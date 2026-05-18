package segmentacion;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import spark.Spark;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class ApiServer {
    private static SegmentadorWeka segmentador;
    private static Map<String, Object> clusterInfo;
    private static Map<String, Object> puntosPorCluster;
    private static PcaParams pcaParams;

    // ─── Estructura de pca_params.json ───────────────────────────────────────
    static class PcaParams {
        boolean usa_scaler;   // true si se usó StandardScaler antes del PCA
        double[] scaler_mean; // media por columna del StandardScaler
        double[] scaler_std;  // desv. estándar por columna del StandardScaler
        double[] media;       // pca.mean_ (media en espacio escalado)
        double[][] componentes; // pca.components_[:2]
    }

    public static void main(String[] args) throws Exception {
        segmentador = new SegmentadorWeka();

        try (Reader r = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/cluster_info.json"), StandardCharsets.UTF_8)) {
            Type type = new TypeToken<Map<String, Object>>(){}.getType();
            clusterInfo = new Gson().fromJson(r, type);
        }

        try (Reader r = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/puntos_por_cluster.json"), StandardCharsets.UTF_8)) {
            Type type = new TypeToken<Map<String, Object>>(){}.getType();
            puntosPorCluster = new Gson().fromJson(r, type);
        }

        try (Reader r = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/pca_params.json"), StandardCharsets.UTF_8)) {
            pcaParams = new Gson().fromJson(r, PcaParams.class);
        }

        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        Spark.port(port);
        Spark.staticFiles.location("/web");

        Spark.get("/puntos", (req, res) -> {
            res.type("application/json");
            return new Gson().toJson(puntosPorCluster);
        });

        Spark.post("/segmentar", (req, res) -> {
            res.type("application/json");
            Gson gson = new Gson();
            Map<String, Object> entrada = gson.fromJson(req.body(), Map.class);

            double montoImp = ((Number) entrada.get("montoImpuesto")).doubleValue();
            double montoPag = ((Number) entrada.get("montoPagado")).doubleValue();
            if (montoImp > 0 && montoPag > montoImp) {
                return gson.toJson(Map.of("error", "El monto pagado no puede superar al impuesto."));
            }

            // Vector de 10 atributos para K-Means (igual que en entrenamiento)
            double[] valores = new double[10];
            valores[0] = ((Number) entrada.get("tipoContribuyente")).doubleValue();
            valores[1] = ((Number) entrada.get("montoImpuesto")).doubleValue();
            valores[2] = ((Number) entrada.get("montoPagado")).doubleValue();
            valores[3] = ((Number) entrada.get("deudaAcumulada")).doubleValue();
            valores[4] = ((Number) entrada.get("aniosMoroso")).doubleValue();
            valores[5] = ((Number) entrada.get("frecuenciaPago")).doubleValue();
            valores[6] = ((Number) entrada.get("mesesRetrasoPromedio")).doubleValue();
            valores[7] = ((Number) entrada.get("numNotificaciones")).doubleValue();
            valores[8] = ((Number) entrada.get("respondioNotificacion")).doubleValue();
            valores[9] = ((Number) entrada.get("fraccionamiento")).doubleValue();

            int cluster = segmentador.predecir(valores);

            // Proyectar con el pipeline completo (Scaler → PCA)
            double[] proyectado = proyectarPCA(valores);

            Map<String, Object> info = (Map<String, Object>) clusterInfo.get(String.valueOf(cluster));
            if (info == null) info = Map.of("nombre", "Segmento " + cluster);

            Map<String, Object> respuesta = new HashMap<>();
            respuesta.put("cluster", cluster);
            respuesta.put("info", info);
            respuesta.put("pcaX", proyectado[0]);
            respuesta.put("pcaY", proyectado[1]);
            return gson.toJson(respuesta);
        });

        Spark.exception(Exception.class, (e, req, res) -> {
            res.status(400);
            res.type("application/json");
            res.body(new Gson().toJson(Map.of("error", e.getMessage())));
        });

        System.out.println("Servidor iniciado en puerto " + port);
    }

    /**
     * Aplica el mismo pipeline que Python:
     *   1) StandardScaler (si usa_scaler=true): x_scaled = (x - scaler_mean) / scaler_std
     *   2) Centrar con pca.mean_:               x_cent   = x_scaled - media
     *   3) Proyectar con componentes PCA:        result   = componentes @ x_cent
     */
    private static double[] proyectarPCA(double[] vectorOriginal) {
        int n = vectorOriginal.length;
        double[] entrada = new double[n];

        // Paso 1: StandardScaler (solo si se usó en entrenamiento)
        if (pcaParams.usa_scaler
                && pcaParams.scaler_mean != null
                && pcaParams.scaler_std  != null) {
            for (int i = 0; i < n; i++) {
                double std = pcaParams.scaler_std[i];
                entrada[i] = (std > 1e-10)
                        ? (vectorOriginal[i] - pcaParams.scaler_mean[i]) / std
                        : 0.0;
            }
        } else {
            // Sin scaler: usar valores crudos
            System.arraycopy(vectorOriginal, 0, entrada, 0, n);
        }

        // Paso 2: Centrar con pca.mean_
        double[] centrado = new double[n];
        for (int i = 0; i < n; i++) {
            centrado[i] = entrada[i] - pcaParams.media[i];
        }

        // Paso 3: Proyectar en las 2 primeras componentes
        double[] resultado = new double[2];
        for (int i = 0; i < 2; i++) {
            double suma = 0;
            for (int j = 0; j < n; j++) {
                suma += pcaParams.componentes[i][j] * centrado[j];
            }
            resultado[i] = suma;
        }
        return resultado;
    }
}

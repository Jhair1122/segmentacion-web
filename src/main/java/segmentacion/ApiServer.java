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

    static class PcaParams {
        double[] media;
        double[][] componentes;
    }

    public static void main(String[] args) throws Exception {
        segmentador = new SegmentadorWeka();

        // Cargar cluster_info.json
        try (Reader reader = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/cluster_info.json"), StandardCharsets.UTF_8)) {
            Type type = new TypeToken<Map<String, Object>>(){}.getType();
            clusterInfo = new Gson().fromJson(reader, type);
        }

        // Cargar puntos_por_cluster.json
        try (Reader reader = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/puntos_por_cluster.json"), StandardCharsets.UTF_8)) {
            Type type = new TypeToken<Map<String, Object>>(){}.getType();
            puntosPorCluster = new Gson().fromJson(reader, type);
        }

        // Cargar parámetros PCA
        try (Reader reader = new InputStreamReader(
                ApiServer.class.getResourceAsStream("/pca_params.json"), StandardCharsets.UTF_8)) {
            pcaParams = new Gson().fromJson(reader, PcaParams.class);
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

            // Construir vector de 10 atributos (igual que en SegmentadorWeka)
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

            // Proyectar el punto usando PCA
            double[] proyectado = proyectarPCA(valores);
            double pcaX = proyectado[0];
            double pcaY = proyectado[1];

            Map<String, Object> info = (Map<String, Object>) clusterInfo.get(String.valueOf(cluster));
            if (info == null) info = Map.of("nombre", "Segmento " + cluster);

            Map<String, Object> respuesta = new HashMap<>();
            respuesta.put("cluster", cluster);
            respuesta.put("info", info);
            respuesta.put("pcaX", pcaX);
            respuesta.put("pcaY", pcaY);
            return gson.toJson(respuesta);
        });

        Spark.exception(Exception.class, (e, req, res) -> {
            res.status(400);
            res.type("application/json");
            res.body(new Gson().toJson(Map.of("error", e.getMessage())));
        });

        System.out.println("Servidor de segmentación (Weka) iniciado en puerto " + port);
    }

    private static double[] proyectarPCA(double[] vectorOriginal) {
        // Centrar el vector (restar la media)
        double[] centrado = new double[vectorOriginal.length];
        for (int i = 0; i < vectorOriginal.length; i++) {
            centrado[i] = vectorOriginal[i] - pcaParams.media[i];
        }
        // Proyectar en las dos primeras componentes
        double[] resultado = new double[2];
        for (int i = 0; i < 2; i++) {
            double suma = 0;
            for (int j = 0; j < vectorOriginal.length; j++) {
                suma += pcaParams.componentes[i][j] * centrado[j];
            }
            resultado[i] = suma;
        }
        return resultado;
    }
}
package segmentacion;

import weka.clusterers.SimpleKMeans;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import java.io.ObjectInputStream;

public class SegmentadorWeka {
    private SimpleKMeans kmeans;
    private Instances estructura;

    public SegmentadorWeka() throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(
                getClass().getResourceAsStream("/modelo_kmeans.model"))) {
            kmeans = (SimpleKMeans) ois.readObject();
        }
        try (java.io.InputStream is = getClass().getResourceAsStream("/segmentacion_estructura.arff")) {
            Instances temp = new Instances(new java.io.InputStreamReader(is));
            estructura = new Instances(temp, 0);
        }
    }

    public int predecir(double[] valores) throws Exception {
        Instance inst = new DenseInstance(1.0, valores);
        inst.setDataset(estructura);
        return kmeans.clusterInstance(inst);
    }
}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

public class Network {
    private static final DataReader dataReader = new DataReader();
    private static final MultiLayerNetwork network = new MultiLayerNetwork(
            new NetworkConfiguration().getMultiLayerConfiguration());

    public void start() {
        network.init();

        performLearning();

        System.out.print(evaulateTestData().stats());
    }

    private void performLearning() {
        CollectScoresIterationListener collectScoresItertionListener = new CollectScoresIterationListener();

        network.addListeners(collectScoresItertionListener);

        for (int i = 0; i < NetworkProperties.ITERATIONS.property; i++) {
            network.fit(dataReader.getTrainingData());
        }

        try {
            collectScoresItertionListener.exportScores(new File("errorFunctionValues.txt"), ",");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Evaluation evaulateTestData() {
        INDArray output = network.output(dataReader.getTestData().getFeatures());
        Evaluation eval = new Evaluation(NetworkProperties.OUTPUTS.property);
        eval.eval(dataReader.getTestData().getLabels(), output);

        return eval;
    }


}

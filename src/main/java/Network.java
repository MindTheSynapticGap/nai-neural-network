import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class Network {

    public void start() {

        final MultiLayerNetwork network = new MultiLayerNetwork(new NetworkConfiguration().getMultiLayerConfiguration());
        network.init();

        final int eachIterations = 50;
        network.addListeners(new ScoreIterationListener(eachIterations));

        DataReader dataReader = new DataReader();
        DataSet trainingData = dataReader.getTrainingData();
        DataSet testData = dataReader.getTestData();

        for(int i = 0; i < NetworkProperties.ITERATIONS.property; i++) {
            network.fit(trainingData);
        }

        INDArray output = network.output(testData.getFeatures());
        Evaluation eval = new Evaluation(NetworkProperties.OUTPUTS.property);
        eval.eval(testData.getLabels(), output);

        System.out.print(eval.stats());
    }

}

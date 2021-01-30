import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
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

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        network.setListeners(new StatsListener(statsStorage));

        performLearning();

        System.out.print(evaulateTestData().stats());

        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


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

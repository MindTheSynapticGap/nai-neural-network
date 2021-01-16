import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkConfiguration {

    private final MultiLayerConfiguration multiLayerConfiguration;

    public NetworkConfiguration() {

        multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SIGMOID)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(NetworkProperties.INPUTS.property)
                        .nOut(NetworkProperties.HIDDEN_NODES.property)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(NetworkProperties.HIDDEN_NODES.property)
                        .nOut(NetworkProperties.OUTPUTS.property)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();
    }

    public MultiLayerConfiguration getMultiLayerConfiguration() {
        return multiLayerConfiguration;
    }

}

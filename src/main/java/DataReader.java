import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Optional;

public class DataReader {

    private final DataSet trainingData;
    private final DataSet testData;

    public DataReader() {
        DataSet allData = populateAllData().orElseThrow(() -> new RuntimeException("Test data could not be populated"));

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70);
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();
    }

    public DataSet getTrainingData() {
        trainingData.shuffle(42);
        return trainingData;
    }

    public DataSet getTestData() {
        return testData;
    }

    private Optional<DataSet> populateAllData() {
        DataSet data = null;

        try (RecordReader recordReader = new CombiningCSVRecordReader(7,0, ',')) {

            recordReader.initialize(new FileSplit(new ClassPathResource("neural_network_data.csv").getFile()));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,
                    NetworkProperties.BATCH_SIZE.property, 30, NetworkProperties.OUTPUTS.property);
            data = iterator.next();


        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }

        if (data != null) {
            return Optional.of(data);
        } else {
            return Optional.empty();
        }
    }


}

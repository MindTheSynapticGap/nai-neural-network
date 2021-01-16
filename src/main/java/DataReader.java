import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class DataReader {

    private DataSet trainingData;
    private DataSet testData;

    public DataReader() {
        try (
                RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("neural_network_data.csv").getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, NetworkProperties.BATCH_SIZE.property, 30, NetworkProperties.OUTPUTS.property);
            DataSet allData = iterator.next();
            allData.shuffle(42);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70);
            trainingData = testAndTrain.getTrain();
            testData = testAndTrain.getTest();

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public DataSet getTrainingData() {
        return trainingData;
    }

    public DataSet getTestData() {
        return testData;
    }
}

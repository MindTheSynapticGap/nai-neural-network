import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import java.util.ArrayList;
import java.util.List;


public class CombiningCSVRecordReader extends CSVRecordReader {
    private final int combineNLines;

    public CombiningCSVRecordReader(int labelLineIndex, int skipNumLines, char delimiter) {
        super(skipNumLines, delimiter);
        this.combineNLines = labelLineIndex;
    }

    @Override
    public List<Writable> next() {
        List<Writable> resultList = new ArrayList<>();
        for(int i=0; i<combineNLines && hasNext(); i+=1) {
            List<Writable> result = super.next();
            resultList.addAll(result);
        }

        return resultList;
    }
}

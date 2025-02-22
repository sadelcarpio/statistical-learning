package data;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Dataset {

    private final float[][] predictors;
    private final float[] labels;

    public float[][] getPredictors() {
        return predictors;
    }

    public float[] getLabels() {
        return labels;
    }

    public Dataset(String path) {
        FileReader fr;
        try {
            fr = new FileReader(path);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        CSVReader reader = new CSVReaderBuilder(fr).build();
        List<String[]> allData;
        try {
            allData = reader.readAll();
        } catch (IOException | CsvException e) {
            throw new RuntimeException(e);
        }

        predictors = new float[allData.size()][];
        labels = new float[allData.size()];

        for (int i = 0; i < allData.size(); i++) {
            String[] row = allData.get(i);
            predictors[i] = new float[row.length - 1];
            for (int j = 0; j < row.length; j++) {
                if ( j == row.length - 1) {
                    labels[i] = Integer.parseInt(row[j]);
                }
                else {
                    predictors[i][j] = Float.parseFloat(row[j]);
                }
            }
        }
    }
}

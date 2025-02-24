package data;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Dataset {

    private final double[][] predictors;
    private final double[][] labels;

    public double[][] getPredictors() {
        return predictors;
    }

    public double[][] getLabels() {
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

        predictors = new double[allData.size()][];
        labels = new double[allData.size()][];
        Set<Integer> labelSet = new HashSet<>();

        for (String[] row : allData) {
            int label = Integer.parseInt(row[row.length - 1]);
            labelSet.add(label);
        }

        int numLabels = labelSet.size();

        for (int i = 0; i < allData.size(); i++) {
            String[] row = allData.get(i);
            predictors[i] = new double[row.length - 1];
            labels[i] = new double[numLabels];
            for (int j = 0; j < row.length; j++) {
                if ( j == row.length - 1) {
                    int label = Integer.parseInt(row[j]);
                    labels[i][label] = 1;
                }
                else {
                    predictors[i][j] = Float.parseFloat(row[j]);
                }
            }
        }
    }
}

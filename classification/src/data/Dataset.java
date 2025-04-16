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

/**
 * Class for classification dataset.
 */
public class Dataset {

    /**
     * Array of predictors of shape {@code [n][p]}, where {@code predictors[i][j]} is the
     * {@code j_th} predictor value for the {@code i_th} example.
     * All must be numeric values (double)
     */
    private final double[][] predictors;
    /**
     * Array of labels of shape {@code [n][K]}, where {@code labels[i][k]} is the {@code k_th} class probability (or
     * logit) for the {@code i_th} example.
     */
    private final double[][] labels;
    public int numClasses;
    public int numPredictors;

    public double[][] getPredictors() {
        return predictors;
    }

    public double[][] getLabels() {
        return labels;
    }

    /**
     * @param path Path to .csv file (without headers) containing the values of predictors, and label in the last column
     *             (as an integer)
     */
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

        numClasses = labelSet.size();

        for (int i = 0; i < allData.size(); i++) {
            String[] row = allData.get(i);
            predictors[i] = new double[row.length - 1];
            labels[i] = new double[numClasses];
            for (int j = 0; j < row.length; j++) {
                if (j == row.length - 1) {
                    int label = Integer.parseInt(row[j]);
                    labels[i][label] = 1;
                } else {
                    predictors[i][j] = Float.parseFloat(row[j]);
                }
            }
        }
        numPredictors = predictors[0].length;
    }
}

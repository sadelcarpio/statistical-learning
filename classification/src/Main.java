import data.Dataset;
import models.discriminative.LogisticRegressionClassifier;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Dataset dataset = new Dataset("data/dummy_data_classification.csv");
        System.out.println(Arrays.toString(dataset.getLabels()));
        System.out.println(Arrays.deepToString(dataset.getPredictors()));
        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
        classifier.fit(dataset);
    }
}

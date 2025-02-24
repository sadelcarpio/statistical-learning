import data.Dataset;
import metrics.Accuracy;
import metrics.Metric;
import models.discriminative.LogisticRegressionClassifier;

import java.util.Arrays;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Dataset dataset = new Dataset("data/dummy_data_classification.csv");
        System.out.println(Arrays.deepToString(dataset.getLabels()));
        System.out.println(Arrays.deepToString(dataset.getPredictors()));
        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier(3);
        classifier.fit(dataset);
        Map<String, Double> metrics = classifier.evaluate(dataset, new Metric[]{new Accuracy()});
        System.out.println(metrics);
    }
}

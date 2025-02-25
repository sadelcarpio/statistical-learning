import data.Dataset;
import metrics.Accuracy;
import metrics.Metric;
import models.discriminative.LogisticRegressionClassifier;
import models.generative.parametric.LDAClassifier;
import models.generative.parametric.QDAClassifier;

import java.util.Arrays;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Metric[] metrics = new Metric[]{new Accuracy()};
        Dataset dataset = new Dataset("data/dummy_data_classification.csv");
        System.out.println("Labels: " + Arrays.deepToString(dataset.getLabels()));
        System.out.println("Predictors: " + Arrays.deepToString(dataset.getPredictors()));
        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier(dataset.numClasses);
        classifier.fit(dataset);
        Map<String, Double> LogRegmetrics = classifier.evaluate(dataset, metrics);
        System.out.println(LogRegmetrics);
        LDAClassifier lda = new LDAClassifier(dataset.numClasses);
        lda.fit(dataset);
        Map<String, Double> LDAMetrics = lda.evaluate(dataset, metrics);
        System.out.println(LDAMetrics);

        QDAClassifier qda = new QDAClassifier(dataset.numClasses);
        qda.fit(dataset);
        Map<String, Double> QDAMetrics = qda.evaluate(dataset, metrics);
        System.out.println(QDAMetrics);
    }
}

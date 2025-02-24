import data.Dataset;
import models.generative.GenerativeClassifier;
import models.generative.LDAClassifier;

public class Main {
    public static void main(String[] args) {
        Dataset dataset = new Dataset("data/dummy_data_classification.csv");
//        System.out.println(Arrays.deepToString(dataset.getLabels()));
//        System.out.println(Arrays.deepToString(dataset.getPredictors()));
//        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier(3);
//        classifier.fit(dataset);
//        Map<String, Double> metrics = classifier.evaluate(dataset, new Metric[]{new Accuracy()});
//        System.out.println(metrics);
        GenerativeClassifier priorClf = new LDAClassifier(3);
        priorClf.calculatePriorsAndMeans(dataset.getLabels(), dataset.getPredictors());
    }
}

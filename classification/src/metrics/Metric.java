package metrics;

/**
 * Interface for implementing metrics.
 */
public interface Metric {
    /**
     * @param yTrue Ground truth labels (one-hot encoded)
     * @param yFalse Predicted labels (one-hot encoded)
     * @return Metric value
     */
    double calculate(double[][] yTrue, double[][] yFalse);
    String getMetricName();
}

package metrics;

public interface Metric {
    double calculate(double[][] yTrue, double[][] yFalse);
    String getMetricName();
}

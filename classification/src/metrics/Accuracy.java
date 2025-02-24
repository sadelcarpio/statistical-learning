package metrics;

import utils.ArrayUtils;

public class Accuracy implements Metric {
    public double calculate(double[][] yTrue, double[][] yHat) {
       int totalPreds = yTrue.length;
       int correctPreds = 0;
       for (int i = 0; i < totalPreds; i++) {
            int argMaxTrue = ArrayUtils.getArgmax(yTrue[i]).get(0);
            int argMaxPred = ArrayUtils.getArgmax(yHat[i]).get(0);
            if (argMaxTrue == argMaxPred) {
                correctPreds++;
            }
       }
       return (double) correctPreds / (double) totalPreds;
    }

    public String getMetricName() {
        return "Accuracy";
    }
}

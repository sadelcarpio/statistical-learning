package models.generative;

import data.Dataset;
import org.ejml.simple.SimpleMatrix;

public class QDAClassifier extends GenerativeClassifier {
    public QDAClassifier(int nClasses) {
        super(nClasses);
    }

    @Override
    public void fit(Dataset dataset) {

    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {

    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        return new double[0][];
    }
}

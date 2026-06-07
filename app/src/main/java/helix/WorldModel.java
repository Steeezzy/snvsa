package helix;

import jdk.incubator.vector.*;

public class WorldModel {
    private static final int FULL_DIM = HypervectorOps.DIM; // 10000
    private static final int PROJ_DIM = 1000; // compressed space
    private static final VectorSpecies<Double> SPEC = DoubleVector.SPECIES_256;

    // projection matrices: compress 10k → 1k and back
    private final double[][] encoder; // [PROJ_DIM][FULL_DIM]
    private final double[][] decoder; // [FULL_DIM][PROJ_DIM]

    // world model weights: predict next state in compressed space
    private final double[][] weights; // [PROJ_DIM][PROJ_DIM]

    private double lastPredictionError = 1.0;

    public WorldModel() {
        java.util.Random rng = new java.util.Random(42);
        encoder = new double[PROJ_DIM][FULL_DIM];
        decoder = new double[FULL_DIM][PROJ_DIM];
        weights = new double[PROJ_DIM][PROJ_DIM];

        // random projection (johnson-lindenstrauss)
        double scale = 1.0 / Math.sqrt(PROJ_DIM);
        for (int i = 0; i < PROJ_DIM; i++) {
            for (int j = 0; j < FULL_DIM; j++) {
                encoder[i][j] = rng.nextGaussian() * scale;
                decoder[j][i] = encoder[i][j]; // transpose
            }
            for (int j = 0; j < PROJ_DIM; j++) {
                weights[i][j] = rng.nextGaussian() * 0.3; // important: 0.3 not 0.01
            }
        }
    }

    // compress a full HV down to 1000 dims
    private double[] encode(int[] hv) {
        double[] out = new double[PROJ_DIM];
        for (int i = 0; i < PROJ_DIM; i++) {
            double sum = 0;
            for (int j = 0; j < FULL_DIM; j++) {
                sum += encoder[i][j] * hv[j];
            }
            out[i] = Math.tanh(sum); // squash
        }
        return out;
    }

    // predict next compressed state from current
    private double[] predict(double[] compressed) {
        double[] next = new double[PROJ_DIM];
        for (int i = 0; i < PROJ_DIM; i++) {
            double sum = 0;
            for (int j = 0; j < PROJ_DIM; j++) {
                sum += weights[i][j] * compressed[j];
            }
            next[i] = Math.tanh(sum);
        }
        return next;
    }

    // update weights when we see what actually happened
    private void updateWeights(double[] predicted, double[] actual, double lr) {
        for (int i = 0; i < PROJ_DIM; i++) {
            double error = actual[i] - predicted[i];
            for (int j = 0; j < PROJ_DIM; j++) {
                weights[i][j] += lr * error * predicted[j];
            }
        }
    }

    // compute prediction error between two compressed vectors
    private double computeError(double[] predicted, double[] actual) {
        double sum = 0;
        for (int i = 0; i < PROJ_DIM; i++) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum / PROJ_DIM);
    }

    private int[] lastSeenHV = null;

    // main method: call this every tick
    // returns prediction error (0 = perfect, 1+ = very wrong)
    public double tick(int[] currentHV, int[] previousHV, double envReward) {
        if (previousHV == null) return 1.0;

        // stability check: only update when state changes meaningfully
        if (lastSeenHV != null) {
            double stability = HypervectorOps.similarity(currentHV, lastSeenHV);
            if (stability > 0.70) return lastPredictionError; // state stable, skip update
        }
        lastSeenHV = currentHV.clone();

        double[] prevCompressed = encode(previousHV);
        double[] currCompressed = encode(currentHV);
        double[] predicted      = predict(prevCompressed);

        double error = computeError(predicted, currCompressed);
        double improvement = lastPredictionError - error;

        double lr = 0.01 + (envReward > 0 ? 0.05 : 0.0);
        updateWeights(predicted, currCompressed, lr);

        lastPredictionError = error;
        lastImprovement = improvement;
        return error;
    }

    private double lastImprovement = 0.0;
    public double getLastImprovement() { return lastImprovement; }

    public double getLastPredictionError() {
        return lastPredictionError;
    }
}

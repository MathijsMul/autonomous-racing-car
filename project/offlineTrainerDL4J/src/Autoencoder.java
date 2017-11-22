import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Autoencoder {
    //autoencoder reduces dimensionality of sensors

    public static final int seed = 12345;
    public static final int iterations = 1;

    public static void main(String[] args) throws  Exception {
        //read csv file
        String csvFile = "aalborg.csv";
        String line = "";
        ArrayList<String[]> tickData = new ArrayList<String[]>();

        try (BufferedReader br = new BufferedReader(new FileReader(new ClassPathResource(csvFile).getFile()))) {
            while ((line = br.readLine()) != null) {
                String[] entries = line.split(",");
                tickData.add(entries);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        tickData.remove(0);//remove heading
        tickData.remove(tickData.size() - 1);//remove last incomplete line

        //convert to double[][]/RealMatrix to allow column-wise actions
        double[][] data = new double[tickData.size()][tickData.get(0).length];
        for (int i = 0; i < tickData.size(); i++) {
            data[i] = Arrays.stream(tickData.get(i))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
        }
        RealMatrix rm = new Array2DRowRealMatrix(data);

        //Create DataSet
        INDArray outPut = Nd4j.create(rm.getColumn(0));
        for (int i = 1; i < 3; i++) {
            outPut = Nd4j.hstack(outPut, Nd4j.create(rm.getColumn(i)));
        }
        INDArray inPut = Nd4j.create(rm.getColumn(3));
        for (int i = 4; i < rm.getColumnDimension(); i++) {
            inPut = Nd4j.hstack(inPut, Nd4j.create(rm.getColumn(i)));
        }
        DataSet dataSet = new DataSet(inPut, outPut);
        List<DataSet> listDs = dataSet.asList();
        DataSetIterator iterator = new ListDataSetIterator(listDs, tickData.size());

        //autoencoder
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(dataSet.numInputs()).nOut(25).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(1, new RBM.Builder().nIn(25).nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(2, new RBM.Builder().nIn(10).nOut(5).lossFunction(LossFunctions.LossFunction.MSE).build())

                //encoding stops, decoding starts
                .layer(3, new RBM.Builder().nIn(5).nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(4, new RBM.Builder().nIn(10).nOut(25).lossFunction(LossFunctions.LossFunction.MSE).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(25).nOut(dataSet.numInputs()).build())
                .pretrain(true).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(iterations / 5)));

        //train model
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            model.fit(new DataSet(next.getFeatures(), next.getFeatures()));
        }

        //Get weights
        org.deeplearning4j.nn.api.Layer layer0 = model.getLayer(0);
        org.deeplearning4j.nn.api.Layer layer1 = model.getLayer(1);
        INDArray[] weightMatrices = new INDArray[] {
                layer0.getParam(DefaultParamInitializer.WEIGHT_KEY),
                layer1.getParam(DefaultParamInitializer.WEIGHT_KEY),
        };
        INDArray[] hiddenBiases = new INDArray[] {
                layer0.getParam(DefaultParamInitializer.BIAS_KEY),
                layer1.getParam(DefaultParamInitializer.BIAS_KEY),
        };

        //TO DO: test performance
        //To DO: make 2 neuralnets; encoder + decoder using above structure/weights to check if encoder works
        //TO DO: save encoder for use in Torcs & offlineTrainer

    }
}

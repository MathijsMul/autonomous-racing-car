import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.io.File;

public class offlineTrainerDL4J {
    public static final int popSize = 10;
    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of iterations per minibatch
    public static final int iterations = 1;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 1480;
    //Network learning rate
    public static final double learningRate = 0.005;

    public static final Random rng = new Random(seed);
    private static Logger log = LoggerFactory.getLogger(offlineTrainerDL4J.class);

    public static void main(String[] args) throws  Exception {
        //read csv file
        String csvFile = "test.csv";
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
        tickData.remove(tickData.size()-1);//remove last incomplete line

        //convert to double[][]/RealMatrix to allow column-wise actions
        double[][] data = new double[tickData.size()][tickData.get(0).length];
        for (int i=0; i< tickData.size(); i++){
            data[i] = Arrays.stream(tickData.get(i))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
        }
        RealMatrix rm = new Array2DRowRealMatrix(data);

         //Create DataSet
        INDArray outPut = Nd4j.create(rm.getColumn(0), new int[]{rm.getRowDimension(),1});//ACCELERATION
        outPut = outPut.sub(Nd4j.create(rm.getColumn(1), new int[]{rm.getRowDimension(),1}));//sub BRAKE, result -1 to 1 (<0 brake, >0 accelerate)
        INDArray inPut = Nd4j.create(rm.getColumn(2), new int[]{rm.getRowDimension(),1});//SPEED
        for (int i=3; i<=7; i++){//TRACK_POSITION,ANGLE_TO_TRACK_AXIS,RADIUS,CORNER_DIRECTION,TRACKEDGE_9
            inPut = Nd4j.hstack(inPut,Nd4j.create(rm.getColumn(i), new int[]{rm.getRowDimension(),1}));
        }
        DataSet dataSet = new DataSet(inPut, outPut);

        MultiLayerNetwork net = null;
        for (int individual=0; individual<popSize; individual++) {
            int iterSeed = seed + individual;
            //Create the network
            int numInput = 6;
            int numOutputs = 1;
            int nHidden = 10;
            net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(iterSeed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)//alternatively: DenseLayer, GravesLSTM
                    .activation("tanh")
                    .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)//alternatively: OutputLayer, RnnOutputLayer
                    .activation("identity")
                    .nIn(nHidden).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()
            );
            net.init();
            net.setListeners(new ScoreIterationListener(1));

            //Train the network on the full data set, and evaluate it periodically
            for( int i=0; i<nEpochs; i++ ){
                 net.fit(dataSet);
            }

            //Get weights + biases
            INDArray weights = net.params();
            //save weights to txt
            String weightsLine = weights.data().toString();
            weightsLine = weightsLine.substring(1,weightsLine.length()-2);//remove '[' and ']' from string
            try (java.io.FileWriter outfile = new java.io.FileWriter("population.txt", individual==0?false:true)) {
                outfile.write(weightsLine+"\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        //Save the model
        File locationToSave = new File("MyMultiLayerNetwork.zip");
        boolean saveUpdater = false;
        try {
            ModelSerializer.writeModel(net, locationToSave, saveUpdater);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

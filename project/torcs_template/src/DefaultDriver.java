import cicontest.algorithm.abstracts.AbstractDriver;
import cicontest.algorithm.abstracts.DriversUtils;
import cicontest.torcs.controller.extras.*;
import cicontest.torcs.genome.IGenome;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import cicontest.algorithm.abstracts.map.TrackMap;
import cicontest.torcs.controller.Driver;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scr.Action;
import scr.SensorModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import static java.lang.Double.isFinite;

//TO DO: clean up

public class DefaultDriver extends AbstractDriver {

    private MultiLayerNetwork nnet;
    private boolean boolOutput = false;
    private static double STEERLOCK = Math.PI / 4; //maximum steering
    private static double LOOKAHEAD_CONST = 12.5; //Lookahead = LOOKAHEAD_CONST + Speed*LOOKAHEAD_FACTOR
    private static double LOOKAHEAD_FACTOR = 0.25; //Lookahead = LOOKAHEAD_CONST + Speed*LOOKAHEAD_FACTOR
    private boolean recoverLeft = true;
    private boolean recoveryMode = false;
    private int trackWidth = 0;
    private int trackDistance = 0;
    private double bestLap = 0;
    private int completedLaps = 0;
    private boolean updateNeeded = false;
    private String controller = "EROAD";
    double stuckTime=9999999;
    double recoveredTime=9999999;
    boolean recovering=false;
    private static double[] ROAD = {0.11393896, 0.17480123, 0.07567492, 0.31979972, 0.06510664, 0.22100267, 0.09878644, -0.3118332825, 0.307170846, -0.25516105, 0.21301426, -0.02567468699999999995, -0.358288783, -0.918397845499999999, 0.011567806, -0.2868883, -0.26082334, -0.6571384, 0.2717822, -0.2800475269999999996, 0.20045806, -0.04626187000000001, 0.12886977, -0.64891165, 0.354283045, -0.09482662, -0.043187275000000025, 0.0523455, 0.10469058, 0.279399245, -0.09968778500000003, 0.313894, 0.6774763, 0.2954692, -0.68283176, -0.20528609, -0.2524819349999999994, 0.32809144, -0.1316154, 0.47842458, 0.017715617500000003, 0.2962551, 0.15944195, -0.26841623, 0.30741748, 0.086315446, 0.37619928, -0.54929954, -1.02776621, 0.23799388, -0.314495675, -0.42715856, -0.43639463, 0.19889386, -0.20507316625, -0.27164617, 0.3474877, 0.05989024, -0.038673423, 0.16486403, -0.10228049075881, -0.07212158, -1.9123723E-7, 0.010315545, 0.0, 3.0172471E-9, -8.71558649999999999E-4, 0.17703211, -2.9291518E-4, 0.0, -0.44957104, 0.468175596, -0.816981334999999999, 0.0812179454999999999, 0.27237079, 0.25340235, 0.12711133, -0.5500691, -0.19847694, 0.09804597, 0.239801};
    private static double[] DIRT = {0.1515648, -0.64431834, -0.23201314, -0.0313187, 0.5126673, -0.41167346, -0.035123542, 0.09054339, -0.18236843, -0.08830849, 0.17667565, -0.1971356, -0.31021866, 0.08446868, -0.47355664, -0.07248625, -0.11247451, -0.1409825, -0.054367065, 0.15452239, 0.4187315, 0.6022131, -0.120444156, 0.25390646, -0.17171073, 0.13880971, 0.02426963, -0.37275317, -0.29030755, -0.13691323, 0.17849351, -0.118366055, 0.0292173, 0.24841394, -0.14796492, 0.0071073025, -0.093976416, -0.06469097, 0.03018705, -0.352119, 0.72147655, -0.18235107, 0.19374616, -0.19627617, -0.44757023, 0.37615, 0.2247648, 0.09947531, 0.38006306, 0.03147902, 0.013407839, 0.10603021, -0.05174797, -0.1245899, 0.44577637, 0.16546285, 0.11193693, -0.4947606, -0.11763464, -0.14962094, 0.6549414, -7.0654263E-4, -3.712413E-4, 4.8624087E-4, 0.0, 0.0, 0.0, 0.0, 4.6281028E-4, 0.00874287, -0.47620988, 0.14813863, -0.19670242, -0.38555542, -0.08563108, 0.087752275, -0.4196947, -0.016396035, 0.012355609, -0.2296092, 0.142728};
    private static double[] EROAD = {0.1617998587499999997, -0.18053312437500002, -0.04539569787500003, -0.0983215124999999999, 0.8668201889375001, -0.622071411, -1.0140612976010626, -0.2234770140625, 0.16102526500000003, -0.010783693750000004, 0.017462007750000008, 0.49462418062499997, 0.3332402599999999995, -0.08818558825, -0.984330721, -0.21000903685, -0.2161360695, -0.894027040625, 0.15534655625, -0.14171941875000002, 0.10019656000000002, 0.9657144663749999, 0.12968733300000002, 0.67497701, -0.19275880250000002, 0.268434555, 0.133787645, -0.8747295125000001, -0.07764303250000001, 0.3796723362499999996, 0.32036686031250006, -0.11006255125, -0.30979066875, -0.2117805199999999997, -0.17638711, 0.38016702, 0.09891755087499998, 0.12839371175, 0.21679233111250001, -0.5480971050000001, 0.72147655, -0.4602329420625, 0.295922401249999999, -0.40760803875, -0.07098957037500003, 0.909967430125, 0.26751281787500003, 0.0404590874999999999, 0.02131346187500005, 0.03147901999999999996, 0.5464065495, 0.26214484437499996, -0.13687312000000004, 0.66603928875, 0.269967780990625, 0.155976285, -0.64198848690625, -0.2843607225, -0.52179392, -0.37366739812500005, 0.705780453918035, -0.04942903433884534, 0.13096989195865624, 0.040908524386875006, -0.047887611875, -0.04640383212284411, -3.00644975E-4, -0.088516055, 0.295520040875, -0.0037490118284000075, -0.59507162485, 0.25554638675, -1.24974715, -0.23412934087499992, 0.12046285750000002, 0.05393656475000003, -0.551821135, -0.06562953937499993, -0.012404631937499966, 0.0650854712499999992, 0.09982699999999999999};
    private static double[] ALPINE2 = {-0.1067801074999999998, 0.2880802697499999995, -0.4597275469218749, -0.1882018874999999997, 0.25346818300000007, 0.17485765437500003, 0.6010508850521251, 0.2923510556250001, 0.3058375, 0.4199974465624999, 0.2712566065, -0.20499951925000004, -0.470035447499999999, 0.786553256, -0.693265475, -0.07498401059375001, -0.1170154058999999998, -0.3462278424999999994, 0.15666495062500002, 0.43988194140625003, -0.241803083125, -0.16595654187500003, -0.1937789778499999998, -0.0713883824999999999, -0.38804980275, -0.1600729675, -0.38177792250000003, -0.3828811518750001, 0.24244487375, 0.160522198, -0.4827906959375, -0.7466118268749999, 0.477786931875, -0.8798702620000001, 0.12074998750000004, 0.1374317475, 0.26540990062500003, 0.1938896085, 0.105481387275, -0.474701535, 0.513563300625, -0.394925836, 0.8041573570625, 0.1327375575, 0.5902090524375, 0.29915180637499994, 0.17638219275, 0.15557354762499997, 0.6676994746875, 0.1608268015, 0.089136661, -0.11998211500000003, 0.014228045000000002, 0.1434983774999999997, 0.08933535350156252, -0.17111368706249996, 0.14126839500000002, 0.22479837, -0.08076712, 0.2392682062499999998, -0.65760612045625, 0.008370359983399904, 0.057415822254447496, 0.04652366435, -0.02810756837734375, 0.0, 0.01139405524375, 0.0, 0.02174141511500001, -0.015539813430300004, 0.5010381563249999, 0.39645224175000005, 0.5478661731250001, -0.10244468381249996, -0.6726390362500001, 0.1567864295, -0.011188232500000006, 0.52462613, 0.4170257, -0.08676135125000001, 0.073671002499999999};
    private static double[] OVAL = {0.17329475093750002, -0.018674538984374933, -0.04387475097656246, -0.14992746162499998, 0.6404601727500001, -0.03630572573437496, -0.2837778975179531, 0.1117366394140625, 0.13127797193750004, 0.30882811031249996, 0.4848429192910156, -0.1971356, 0.3619871144375, -0.23507746996093754, -0.3145231409999999995, -0.2661458176828125, -0.43245530435976565, -1.170248423125, -0.35608656250000004, -0.07213256959375008, -0.04363429000000002, -0.3466982714375, -0.07336859201249996, 0.437259734609375, 0.41895753346875003, 0.13880971, -0.045780534359375014, 0.390148124999999999, -0.11385767671874998, -0.22988535809375, -0.1394861699999999996, 0.4907265345312499, 0.5585641459765625, -0.2827632409609375, -0.0032101250000000636, 0.31680144359375007, -0.3117034132499999995, 0.13963872834375002, 0.03953742691875, 0.159456977203125, 0.72147655, -0.058537160375000064, 0.595127890296875, -0.2456152372656249, 0.303894335625, 0.2653087975625, 0.5703566467109376, -0.0441486409999999996, -0.41170065671875006, 0.2388702235625, 0.05258550712500004, 0.10603021, -0.587682815, 0.3189838674999999985, 0.4813515919035157, -0.16991359053125002, 0.36543431742968757, 0.0191643649999999999, -0.7922378913750001, 0.4920898699999999993, 0.39531392572626123, 0.019975524021814822, 0.06683784211611873, -0.01647539796137344, -0.05399787817540625, 6.39376547471E-5, -0.039167942402343746, -0.08171237451390623, 0.0565103770525, 0.010738474961756247, 1.197445590990625, -0.1846937968125, -1.2841097067499998, -0.0356643912499999998, 0.135304698046875, 0.0277801792499999998, -0.59139760825, -0.1293868685546874, 0.65111394003125, 0.56346726828125, 0.203329217};
    private double[] bestWeights = DIRT;
    private double[] currentWeights = ALPINE2;
    private boolean useNN = true;
    private int trackID = -1;

    public DefaultDriver() {
        initialize();
        nnet = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(1)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(6).nOut(10)//alternatively: DenseLayer, GravesLSTM
                        .activation("tanh")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)//alternatively: OutputLayer, RnnOutputLayer
                        .activation("identity")
                        .nIn(10).nOut(1).build())
                .pretrain(false).backprop(true).build()
        );
        nnet.init();
//        /*dirt*/ double[] weights = {0.003572450000000005, -0.450005272, -0.681361, 0.023377763, 0.2596613, 0.22762977, -0.478984189, 0.60853994, 0.3058375, 0.10517079, -0.147665, 0.6136816, -0.20282522, 0.092180006, 0.3494559849999999997, -0.11050891, -0.45494452, -1.015200375, -0.5912853, 0.48455822, -0.040530775000000005, -0.29106665, 0.07420377, 0.2283763475, 0.18677864700000002, -0.28850799, -0.24412537, 0.13363387, 0.11396646, -0.30385743000000004, -0.115060285, 0.02022649500000001, 0.25111654, -0.09474325, 0.5349453, 0.5066353, -0.18976016, -0.19185193, -0.01740871649999999997, -0.43080422, -0.23252232, 0.07273019, 0.40915522, -0.19096465, 0.46926555, 0.1815106229999999998, -0.079859394, 0.08359712, 0.22802219, -0.23079181, 0.32966918, 0.0677723899999999999, 0.06043863500000002, -0.0451497, 0.7948592297218751, 0.169086605, 0.26929328087500004, 0.22479837, -0.08076712, 0.125540695, -0.392663584775, -0.039606735, -0.12003809, -0.004531921, 0.0, 0.0, 0.002560493399999999998, 3.1276137E-4, 0.001604807, 0.08244273, 0.738205435, 0.39662772, 0.5516631, 0.118464395, 0.14321508, 0.075177473, -0.13714154, 0.40356782, 0.4170257, 0.5503509, 0.173709};
//        /*road*/ double[] weights = {-0.03862412, 0.172428325, 0.116306126, -0.18329804, 0.055606265, 0.009122442, -0.17989095, 0.16189447, 0.26299945, -0.26168042, -0.247308, -0.016774878, 0.4051009, -0.012096848, 0.40700642200000003, -0.13566223, 0.0057823057, 0.29196203, 0.09373754, -0.32377319225, -0.32165417, -0.306293, -0.207967015, 0.08963178, -0.1025781439999999998, -0.12043998, -0.1947664, 0.28773471000000006, -0.25707936, -0.09286062, -0.502909, -0.0757196987499999995, 0.1473669, -0.39940894, -0.03427616, -0.2183836, 0.34035006, 0.005607602, -0.0013772269, 0.03983721, -0.21607128750000001, -0.02836091, -0.14585042, -0.25941554, 0.4178461, 0.14203626, -0.028087264, 0.3355402, -0.22142993, 0.0319241, -0.2470615, 0.16034615, -0.09258933, 0.20866628, 3.0251715E-4, -0.029367674, 0.10887791, 0.24594167, 0.6870724, 0.22016996, 6.151762E-8, 0.0, 1.52242355E-5, 0.0068719187, -9.020733E-6, 0.0, 5.04989E-4, -0.08960962528, -0.20482776, -1.3123288E-6, -7.629006E-4, -0.15468508, 0.24828407, 0.4230708, 0.26078054, -0.4021642, 0.19937222, 0.7216795, 0.5033665, -0.54287225, -0.029778};
//        nnet.setParams(Nd4j.create(weights));
        nnet.setParams(Nd4j.create(bestWeights));
    }

    public DefaultDriver(double[] weights, boolean output, boolean useNN) {
        initialize();
        //Load the model
        try {
            nnet = ModelSerializer.restoreMultiLayerNetwork(new ClassPathResource("MyMultiLayerNetwork.zip").getFile());
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.useNN = useNN;
        this.boolOutput = output;
        //Load weights in nnet
        nnet.setParams(Nd4j.create(weights));
    }

    private void initialize() {
        this.enableExtras(new AutomatedClutch());
        this.enableExtras(new AutomatedGearbox());
        //this.enableExtras(new AutomatedRecovering());
        this.enableExtras(new ABS());
/*
        if (boolOutput) {
            try (java.io.FileWriter outfile = new java.io.FileWriter("test.csv", false)) {
                outfile.write("ACCELERATION,BRAKE,SPEED,TRACK_POSITION,ANGLE_TO_TRACK_AXIS,RADIUS,CORNER_DIRECTION,TRACKEDGE_9" + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
*/
    }

    @Override
    public void loadGenome(IGenome genome) {
        if (genome instanceof DefaultDriverGenome) {
            DefaultDriverGenome myGenome = (DefaultDriverGenome) genome;
        } else {
            System.err.println("Invalid Genome assigned");
        }
    }

    @Override
    public double getAcceleration(SensorModel sensors) {
        return 1;
    }

    @Override
    public double getSteering(SensorModel sensors) {
        return 0.5;
    }

    @Override
    public String getDriverName() {
        return "Lightning McQueen";
    }

    @Override
    public Action controlWarmUp(SensorModel sensors) {
        Action action = new Action();
        return defaultControl(action, sensors);
    }

    @Override
    public Action controlQualification(SensorModel sensors) {
        Action action = new Action();
        return defaultControl(action, sensors);
    }

    @Override
    public Action controlRace(SensorModel sensors) {
        Action action = new Action();
        return defaultControl(action, sensors);
    }

    @Override
    public Action determineAction(SensorModel sensors) {
        Action action2 = this.control(sensors);
        int gear = action2.gear;
        Action action = super.determineAction(sensors);
        if (gear == -1) action.gear = -1;
        return action;
    }

    @Override
    public Action defaultControl(Action action, SensorModel sensors) {
        if (boolOutput) System.out.println("--------------" + getDriverName() + "--------------");
        if (action == null) {
            action = new Action();
        }

/*      //test 5 best & diverse controllers to determine best combination of 2
        if ((completedLaps == 1)&&(updateNeeded)){
            nnet.setParams(Nd4j.create(ALPINE2));
            controller="ALPINE2";
            updateNeeded=false;
        }
        if ((completedLaps == 2)&&(updateNeeded)){
            nnet.setParams(Nd4j.create(ROAD));
            controller="ROAD";
            updateNeeded=false;
        }
        if ((completedLaps == 3)&&(updateNeeded)){
            nnet.setParams(Nd4j.create(DIRT));
            controller="DIRT";
            updateNeeded=false;
        }
        if ((completedLaps == 4)&&(updateNeeded)){
            nnet.setParams(Nd4j.create(OVAL));
            controller="OVAL";
            updateNeeded=false;
        }
        if ((completedLaps == 5)&&(updateNeeded)){
            nnet.setParams(Nd4j.create(EROAD));
            controller="EROAD";
            updateNeeded=false;
        }
        if (sensors.getLaps()>completedLaps){
            completedLaps=sensors.getLaps();
            updateNeeded=true;
        }
*/

        if ((sensors.getLaps() == 1)&&(bestLap==0)) {//change from safe controller to aggressive one
            bestLap = sensors.getBestLapTime();
//            currentWeights = OVAL;
            nnet.setParams(Nd4j.create(currentWeights));
        }
        if ((sensors.getLastLapTime() < bestLap)&&(!java.util.Arrays.equals(bestWeights,currentWeights))) {
            bestWeights = currentWeights;//agressive controller is best
        }
        if ((!java.util.Arrays.equals(bestWeights,currentWeights)&&(sensors.getLaps() >= 1))){
            if (sensors.getCurrentLapTime() > bestLap) {//revert to initial controller if current lap is worse than best
                nnet.setParams(Nd4j.create(bestWeights));
                currentWeights = bestWeights;
            }
        }

/*
        System.out.println("Laps: " + sensors.getLaps());
        System.out.println("Best lap " + bestLap);
        System.out.println("Last lap: " + sensors.getLastLapTime());
        System.out.println("Controller: " + controller);
*/

        double axisSpeed = Math.sqrt(Math.pow(sensors.getSpeed(), 2) + Math.pow(sensors.getLateralSpeed(), 2));

        //Detect crash
        if (axisSpeed < 20 && !recoveryMode) {
            recoveryMode = true;
            if (sensors.getTrackPosition() > 0) {
                recoverLeft = true;
            }
        }

        if (recoveryMode) {
            action.brake = 0;
            if ((Math.abs(sensors.getAngleToTrackAxis()) < 6 * Math.PI / 36)||(recovering)) {//car is recovered to correct direction (within -30 and 30 degrees)
                if (Math.abs(sensors.getAngleToTrackAxis()) > 7 * Math.PI / 36) {
                    recovering=false;
                } else recovering=true;
                action.gear = 1;
                action.steering = DriversUtils.alignToTrackAxis(sensors, 1);
                if (Math.abs(sensors.getAngleToTrackAxis()) < Math.PI / 36){//if car is almost completely aligned, full throttle
                    action.accelerate = 1;
                } else action.accelerate = 0.4+(0.6/40)*sensors.getSpeed();
                if (sensors.getSpeed() > 40){
                    recoveryMode = false;
                    recovering=false;
                }
            } else if (recoverLeft) {
                if (sensors.getTrackPosition() < -0.75) {
                    recoverLeft = false;
                } else {
                    if (sensors.getAngleToTrackAxis() < 0) {
                        action.gear = -1;
                        action.steering = 1;
                        action.accelerate = 0.5;
                    } else if (sensors.getAngleToTrackAxis() > 0) {
                        action.gear = 1;
                        action.steering = 1;
                        action.accelerate = 0.5;
                    }
                }
            } else if (!recoverLeft) {
                if (sensors.getTrackPosition() > 0.75) {
                    recoverLeft = true;
                } else {
                    if (sensors.getAngleToTrackAxis() < 0) {
                        action.gear = 1;
                        action.steering = -1;
                        action.accelerate = 0.5;
                    } else if (sensors.getAngleToTrackAxis() > 0) {
                        action.gear = -1;
                        action.steering = -1;
                        action.accelerate = 0.5;
                    }
                }
            }

            if (boolOutput) {
                System.out.println("--------------" + getDriverName() + "--------------");
                System.out.println("Distance: " + sensors.getDistanceFromStartLine());
                System.out.println("Recovery");
                // added angle to track axis:
                System.out.println("Angle to track axis: " + sensors.getAngleToTrackAxis());
                System.out.println("Track Position: " + sensors.getTrackPosition());
                System.out.println("recoverLeft: " + recoverLeft);
                System.out.println("-----------------------------------------------");
            }
        } else {
            //heuristic for steering
            double radius = calculateRadius(sensors);
            int directionOfTurn = directionOfTurn(sensors);
            int longestRangeSensor = findLongestRangeSensor(sensors);
            double lookAhead = LOOKAHEAD_CONST + sensors.getSpeed() * LOOKAHEAD_FACTOR;
            if (sensors.getLateralSpeed() > sensors.getSpeed() * 0.25) {//drifting, align steeer with axis
                action.steering = DriversUtils.alignToTrackAxis(sensors, 1);
            } else if (Math.abs(sensors.getTrackPosition()) > 0.8) { //steer back to middle of track in case car is on the edge
                action.steering = (DriversUtils.moveTowardsTrackPosition(sensors, 1, 0) + DriversUtils.alignToTrackAxis(sensors, 0.5)) / 2;
            } else if (Math.abs(sensors.getAngleToTrackAxis()) < Math.PI / 3) {//in normal situation, steer towards longest point of view
                action.steering = Math.min(1, Math.max(-1, -longestRangeSensor * Math.PI / 18 + Math.PI / 2));
            } else action.steering = DriversUtils.alignToTrackAxis(sensors, 1);//non-normal, correct to align with axis

            if (useNN) {
    //Neural network
                // inputs for prediction function:
                //SPEED,TRACK_POSITION,ANGLE_TO_TRACK_AXIS,RADIUS,CORNER_DIRECTION,TRACKEDGE_9
                double speed = axisSpeed;
                double trackPosition = sensors.getTrackPosition();
                double angleToTrackAxis = sensors.getAngleToTrackAxis();
                double trackEdge_9 = sensors.getTrackEdgeSensors()[9];

                final INDArray input = Nd4j.create(new double[]{speed, trackPosition, angleToTrackAxis, radius, directionOfTurn, trackEdge_9}, new int[]{1, 6});

                // make prediction according to NN
                INDArray out = nnet.output(input, false);

                if (out.getDouble(0) > 0.2) {
                    action.accelerate = 1.0D;
                    action.brake = 0.0D;
                } else if (out.getDouble(0) < -0.2) {
                    action.accelerate = 0.0D;
                    action.brake = 1.0D;
                } else {
                    action.accelerate = 0.0D;
                    action.brake = 0.0D;
                }
                if (boolOutput) System.out.println("NNet: " + out.getDouble(0));
            } else {
                //Heuristics
                if ((sensors.getTrackEdgeSensors()[9] > lookAhead)
                        || (radius < 10)) {
                    action.accelerate = 1.0D;
                    action.brake = 0.0D;
                } else {
                    if (sensors.getSpeed() > Math.max((radius - 10) / 160 * 200, 60) + 5) {//minimum speed 60
                        action.accelerate = 0.0D;
                        action.brake = 1.0D;
                    } else if (sensors.getSpeed() > Math.max((radius - 10) / 160 * 200, 60)) {
                        action.accelerate = 0.0D;
                        action.brake = 0.0D;
                    } else {
                        action.accelerate = 1.0D;
                        action.brake = 0.0D;
                    }
                }
            }


            if (boolOutput) {
                System.out.println("Radius: " + radius);
                System.out.println("Distance: " + sensors.getDistanceFromStartLine());
                System.out.println("Angle to track axis: " + sensors.getAngleToTrackAxis());
                System.out.println("Speed: " + axisSpeed);
                System.out.println("Steering: " + action.steering);
                System.out.println("Acceleration: " + action.accelerate);
                System.out.println("Brake: " + action.brake);
                System.out.println("-----------------------------------------------");
            }

/*
            //save actions + sensor data to csv
            //ACCELERATION,BRAKE,SPEED,TRACK_POSITION,ANGLE_TO_TRACK_AXIS,RADIUS,CORNER_DIRECTION,TRACKEDGE_9
            String line = action.accelerate +","+ action.brake +","+ axisSpeed +","+ sensors.getTrackPosition() +","+ sensors.getAngleToTrackAxis()+","+ radius + "," + directionOfTurn + "," +sensors.getTrackEdgeSensors()[9];
            try (java.io.FileWriter outfile = new java.io.FileWriter("test.csv", true)) {
                outfile.write(line+"\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
*/
        }
        return action;
    }

    private double calculateRadius(SensorModel sensors) {
        //http://www.intmath.com/applications-differentiation/8-radius-curvature.php
        if (Math.abs(sensors.getAngleToTrackAxis()) > Math.PI / 3) {//no calculation when more than 60 degree angle to TrackAxis
            return -1;
        } else {
            int i = findSensorOnAxis(sensors);
            double sensor1 = sensors.getTrackEdgeSensors()[i - 1];//sensor 10 degrees left
            double sensor2 = sensors.getTrackEdgeSensors()[i];//sensor on axis line
            double sensor3 = sensors.getTrackEdgeSensors()[i + 1];//sensor 10 degrees right
            double x1 = Math.sin(-Math.PI / 18) * sensor1;
            double y1 = Math.cos(-Math.PI / 18) * sensor1;
            double x2 = Math.sin(0) * sensor2;
            double y2 = Math.cos(0) * sensor2;
            double x3 = Math.sin(Math.PI / 18) * sensor3;
            double y3 = Math.cos(Math.PI / 18) * sensor3;
            double m1 = (y2 - y1) / (x2 - x1);
            double m2 = (y3 - y2) / (x3 - x2);
            double xc = (m1 * m2 * (y1 - y3) + m2 * (x1 + x2) - m1 * (x2 + x3)) / (2 * (m2 - m1));
            double yc = (-1 / m1) * (xc - (x1 + x2) / 2) + (y1 + y2) / 2;
            return Math.sqrt(Math.pow((x1 - xc), 2) + Math.pow((y1 - yc), 2));
        }
    }

    private int directionOfTurn(SensorModel sensors) {
        int i = findSensorOnAxis(sensors);
        if (Math.abs(sensors.getAngleToTrackAxis()) > Math.PI / 3) {//no calculation when more than 60 degree angle to TrackAxis
            return 99;//not identified
        } else if (sensors.getTrackEdgeSensors()[i] < 100) {
            if (sensors.getTrackEdgeSensors()[i - 1] > sensors.getTrackEdgeSensors()[i + 1] + 10) {
                return -1;//turn to left
            } else if (sensors.getTrackEdgeSensors()[i - 1] + 10 < sensors.getTrackEdgeSensors()[i + 1]) {
                return 1;//turn to right
            }
        }
        return 0;//straight
    }

    private int findSensorOnAxis(SensorModel sensors) {
        return (int) Math.round((sensors.getAngleToTrackAxis() + Math.PI / 2) * 18 / Math.PI);
    }

    private int findLongestRangeSensor(SensorModel sensors) {
        double[] sensorRanges = sensors.getTrackEdgeSensors();
        int longestRangeSensor = 9;// initialize with middle sensor
        for (int i = 0; i < sensorRanges.length; i++) {
            if (sensorRanges[i] > sensorRanges[longestRangeSensor]) longestRangeSensor = i;
        }
        return longestRangeSensor;
    }
}

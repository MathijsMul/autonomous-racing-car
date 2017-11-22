import org.apache.commons.lang.ArrayUtils;
import race.TorcsConfiguration;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DifEvo {

    private static int vector_length = 81;
    private static int popSize;
    public static int evals = 0;
    private static String[][] track = {
            {"mixed-1","dirt"},{"dirt-1","dirt"},{"dirt-5","dirt"},
            {"mixed-2","dirt"},{"dirt-3","dirt"},{"e-track-5","oval"},
            {"dirt-2","dirt"},{"a-speedway","oval"},{"g-track-1","road"},
            {"michigan","oval"},{"aalborg","road"},{"g-track-3","road"},
            {"g-speedway","oval"},{"dirt-6","dirt"},{"g-track-2","road"},
            {"etrack-1","road"},{"eroad","road"},{"dirt-4","dirt"},
            {"ruudskogen","road"},{"c-speedway","oval"},{"d-speedway","oval"},
            {"corkscrew","road"},{"f-speedway","oval"},{"alpine-2","road"},
            {"street-1","road"},{"brondehach","road"},{"b-speedway","oval"},
            {"e-speedway","oval"},{"e-track-3","road"},{"wheel-1","road"},
            {"e-track-6","road"},{"e-track-2","road"},{"forza","road"},
            {"wheel-2","road"},{"ole-road-1","road"},{"alpine-1","road"},
            {"e-track-4","road"},{"spring","road"}};

    public class individual {
        double[] vector = new double[vector_length];
        double fitness_par;

        public individual(double[] vector, double fitness_par) {
            this.vector = vector;
            this.fitness_par = fitness_par;
        }
    }

    public individual[] DE(individual[] pop, int evaluations_limit_, double crossoverRate, int trackID) {
        individual[] newPop = new individual[popSize];
        Random rnd_ = new Random();
        double F = 0.5;
        while (evals < evaluations_limit_) {
            for (int i = 0; i < popSize; i++) {
                if (evals < evaluations_limit_) {
                    individual parent = pop[i];
                    individual offspring = new individual(new double[vector_length], 0);
                    //select 3 random parents
                    int index1, index2, index3;
                    do
                        index1 = rnd_.nextInt(popSize);
                    while (index1 == i);
                    do
                        index2 = rnd_.nextInt(popSize);
                    while ((index2 == index1) || (index2 == i));
                    do
                        index3 = rnd_.nextInt(popSize);
                    while ((index3 == index1) || (index3 == index2) || (index3 == i));
                    individual base = pop[index1];
                    individual ind_1 = pop[index2];
                    individual ind_2 = pop[index3];
                    int crossover_j = rnd_.nextInt(vector_length);
                    for (int j = 0; j < vector_length; j++) {
                        if ((rnd_.nextDouble() <= crossoverRate) || (j == crossover_j)) {
                            offspring.vector[j] = base.vector[j] + F * (ind_1.vector[j] - ind_2.vector[j]);
                        } else {
                            offspring.vector[j] = parent.vector[j];
                        }
                    }
                    offspring.fitness_par = evaluate(offspring, trackID);
                    if (offspring.fitness_par < parent.fitness_par) {
                        newPop[i] = offspring;
                    } else {
                        newPop[i] = parent;
                    }
                }
            }
            pop = newPop;
        }

        return pop;
    }

    public double evaluate(individual ind, int trackID) {
        //Start a race
        DefaultRace race = new DefaultRace();
        race.setTrack(track[trackID][0], track[trackID][1]);
        race.laps = 1;
        //for speedup set withGUI to false
        DefaultDriver[] driversList = new DefaultDriver[1];
        driversList[0] = new DefaultDriver(ind.vector, false, true);
        double[] results = race.runQualification2(driversList, false);
        evals++;
        System.out.println("Evals: " + evals);
        System.out.println("Result: " + results[0]);
        return results[0];
    }

    public individual[] initializePop(int trackID) {
        //String txtFile = "population.txt";
        //String txtFile = "dirt.txt";
        //String txtFile = "oval.txt";
        //String txtFile = "road.txt";
        String txtFile = "combi.txt";
        String line = "";
        ArrayList<String[]> population = new ArrayList<String[]>();

        try (BufferedReader br = new BufferedReader(new FileReader(txtFile))) {
            while ((line = br.readLine()) != null) {
                String[] entries = line.split(",");
                population.add(entries);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        popSize = population.size();
        individual[] pop = new individual[popSize];
        for (int i = 0; i < popSize; i++) {
            individual ind = new individual(new double[vector_length], 0);
            for (int j = 0; j < vector_length; j++) {
                ind.vector[j] = Double.parseDouble(population.get(i)[j]);
            }
            pop[i] = ind;
        }
        int[] dirt = {0,1,2,3,4,6,13,17};
        int[] road = {8,10,11,14,16,18,21,23,24,25,28,29,30,31,32,33,34,35,36,37,};//15 removed
        int[] oval = {5,7,9,12,19,20,22,26,27};
        int[] combi = {23};
        for (int i = 0; i < popSize; i++) {
            //for (int track : dirt)
            //for (int track : road)
            //for (int track : oval)
            for (int track : combi)
                pop[i].fitness_par += evaluate(pop[i],track);
//            pop[i].fitness_par = evaluate(pop[i],trackID);
        }
        return pop;
    }

    public static void main(String[] args) {
        DifEvo test = new DifEvo();
        //Set path to torcs.properties
        TorcsConfiguration.getInstance().initialize(new File("torcs.properties"));

        for (int trackID = 23; trackID< 24/*track.length*/; trackID++) {
            evals = 0;
            individual[] pop = test.initializePop(trackID);
            individual bestPre = pop[0];
            for (int i = 1; i < popSize;i++){
                if (pop[i].fitness_par < bestPre.fitness_par){
                    bestPre = pop[i];
                }
            }
            pop = test.DE(pop, 10000, 0.7, trackID);
            individual best = pop[0];
            for (int i = 1; i < popSize;i++){
                if (pop[i].fitness_par < best.fitness_par){
                    best = pop[i];
                }
            }

            System.out.println("Best pre-evolve: " + bestPre.fitness_par);
            System.out.println("Best evolved: " + best.fitness_par);

            //Start a race with heuristics
            DefaultRace race = new DefaultRace();
            race.setTrack(track[trackID][0], track[trackID][1]);
            race.laps = 1;
            DefaultDriver[] driversList = new DefaultDriver[1];
            driversList[0] = new DefaultDriver(new double[81], false, false);
            double[] results = race.runQualification2(driversList, false);
            double heuristicResult = results[0];

            String weightsLine = Arrays.toString(best.vector);
            weightsLine = weightsLine.substring(1,weightsLine.length()-2);//remove '[' and ']' from string
            try (java.io.FileWriter outfile = new java.io.FileWriter("best "+ trackID + " " + track[trackID][0] + ".txt", false)) {
                outfile.write(weightsLine+"\n");
                for (int i = 0; i < popSize;i++){
                    weightsLine = Arrays.toString(pop[i].vector);
                    weightsLine = weightsLine.substring(1,weightsLine.length()-2);//remove '[' and ']' from string
                    outfile.write(weightsLine+"\n");
                }
                outfile.write("Best pre-evolve: " + bestPre.fitness_par+" Best evolved: " + best.fitness_par+ " Heuristic: " + heuristicResult + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
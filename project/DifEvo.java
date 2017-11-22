/**
 * Created by Xandra Velders on 30-11-2016.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DifEvo {

    private static int vector_length = 10;
    private static int popSize;
    public static int evals =0;

    public class individual {
        double[] vector = new double[vector_length];
        double fitness_par;

        public individual(double[] vector, double fitness_par) {
            this.vector = vector;
            this.fitness_par = fitness_par;
        }
    }

    public individual[] DE(individual[] pop, int evaluations_limit_, double crossoverRate) {
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
                    evaluate(offspring);
                    //offspring.fitness_par = 0;
                    evals++;
                    if (offspring.fitness_par < parent.fitness_par) {
                        newPop[i] = offspring;
                    } else {
                        newPop[i] = parent;
                    }
                }
            }
            pop = newPop;
        }

        return newPop;
    }

    public void evaluate(individual ind) {
        Random rnd_ = new Random();
        ind.fitness_par = rnd_.nextDouble();
    }

    public individual[] initializePop() {
        String csvFile = "test.csv";
        String line = "";
        ArrayList<String[]> tickData = new ArrayList<String[]>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] entries = line.split(",");
                tickData.add(entries);
            }
            //tickData.remove(0);//remove heading
        } catch (IOException e) {
            e.printStackTrace();
        }
        popSize = tickData.size();
        individual[] pop = new individual[popSize];
        for (int i = 0; i < popSize; i++) {
            individual ind = new individual(new double[vector_length], 0);
            for (int j = 0; j < vector_length; j++) {
                ind.vector[j] = Double.parseDouble(tickData.get(i)[j]);
            }
            pop[i] = ind;
        }
        for (int i = 0; i < popSize; i++){
            evaluate(pop[i]);
        }
        return pop;
    }

    public static void main(String[] args) {
        DifEvo test = new DifEvo();
        individual[] pop = test.initializePop();
        pop = test.DE(pop, 5, 0.5);
        for (int i = 0; i < popSize;i++){
            individual ind = pop[i];
            System.out.println("Individual " + i);
            for (int j = 0; j < vector_length; j++){
                System.out.print(ind.vector[j] + " ");
            }
            System.out.println("\nFitness: " + ind.fitness_par + "\n");
        }
    }
}

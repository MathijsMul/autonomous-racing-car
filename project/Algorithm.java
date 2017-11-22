import java.util.Random;
import org.vu.contest.ContestEvaluation;

public class Algorithm {
	int seed;
	private int evals = 0;
	static int POP_SIZE = 100;
	//Memetic
	int budget;
	private double func_factor;
	static double step = 0.03D;
	//DE
	static double CR = 0.2D;
	static double F = 0.5D;

	Selection Selection = new Selection();

	public void run(ContestEvaluation evaluation_, int evaluations_limit_, Random rnd_, int Function)
	{
		if ((Function==1)||(Function==3)){
			seed=1;
			budget = evaluations_limit_;
			func_factor = 0.75;
		} else {
			seed=7;
			budget = 1000;
			func_factor = 0.6;
		}
		rnd_.setSeed(seed);
		Population pop = new Population(POP_SIZE);
		pop = pop.init(rnd_);

		double totalFitness_NM = -999999.0D;
		while (evals<evaluations_limit_){
			double startFitness = totalFitness_NM;
			pop = DE(pop, evaluation_, Math.min((int) (evals+budget*func_factor),evaluations_limit_), CR, rnd_, false);
			double totalFitness_DE = 0.0D;
			if (Function==2){
				for (int i = 0; i < POP_SIZE; i++) {
					totalFitness_DE += pop.getFittest(i).getFitness();
				}
			}
			pop = NelderMead(pop, evaluation_, Math.min((int) (evals+budget*(1-func_factor)),evaluations_limit_), rnd_, Function);
			totalFitness_NM = 0.0D;
			for (int i = 0; i < POP_SIZE; i++) {
				totalFitness_NM += pop.getFittest(i).getFitness();
			}
			if ((totalFitness_DE - startFitness) / (budget * func_factor) > (totalFitness_NM - totalFitness_DE) / (budget * (1.0D - func_factor))) {
				func_factor += step;
			} else {
				func_factor -= step; }
			if (func_factor > 0.9D) func_factor = 0.9D;
			if (func_factor < 0.1D) func_factor = 0.1D;
		}
	}

	public Population evaluate(Population pop, ContestEvaluation evaluation_, int evaluations_limit_) {
		for (int index = 0; index < pop.size(); index++) {
			if ((pop.getIndividual(index).getFitness() == -1.0D) && (evals < evaluations_limit_)) {
				double fitness = ((Double)evaluation_.evaluate(pop.getIndividual(index).getValues())).doubleValue();
				pop.getIndividual(index).setFitness(fitness);
				evals++;
			}
		}
		return pop;
	}

	public Population DE(Population pop, ContestEvaluation evaluation_, int evaluations_limit_, double crossoverRate, Random rnd_, boolean RandSelect) {
		pop = evaluate(pop, evaluation_, evaluations_limit_);
		double F = Algorithm.F;
		Population newPop = new Population(pop.size());
		while (evals < evaluations_limit_) {
			for (int i = 0; i < pop.size(); i++) {
				if (evals < evaluations_limit_) {
					Individual parent = pop.getIndividual(i);
					Individual offspring = new Individual(rnd_);
					// Select 3 random parents
					int index1, index2, index3;
					do 
						index1 = rnd_.nextInt(pop.size());
					while (index1 == i);
					do 
						index2 = rnd_.nextInt(pop.size());
					while ((index2==index1)||(index2 == i));
					do 
						index3 = rnd_.nextInt(pop.size());
					while ((index3==index1)||(index3==index2)||(index3 == i));
					Individual base = pop.getIndividual(index1);
					Individual individual1 = pop.getIndividual(index2);
					Individual individual2 = pop.getIndividual(index3);
					int crossover_j = rnd_.nextInt(10);
					double value;
					for (int j=0; j<10; j++){
						// Apply crossover per allele
						if ((rnd_.nextDouble() <= crossoverRate)||(j==crossover_j)) {
							value = base.getValue(j) + F*(individual1.getValue(j) - individual2.getValue(j));
							value = Math.max(Math.min(value, 5), -5);
						} else {
							value = parent.getValue(j);
						}
						offspring.setValue(j, value);
					}
					//replace parent by offspring in case of higher fitness
					double fitness = (double)evaluation_.evaluate(offspring.getValues());
					evals++;
					if (fitness > parent.getFitness()){
						offspring.setFitness(fitness);
						newPop.setIndividual(i, offspring);
					} else{
						newPop.setIndividual(i, pop.getIndividual(i));
					}
				}
			}
			pop = newPop;
		}
		return newPop;
	}

	public Population NelderMead(Population pop, ContestEvaluation evaluation_, int evaluations_limit_, Random rnd_, int Function) {
		double alpha = 1.0D;
		double gamma = 2.0D;
		double rho = 0.5D;
		double sigma = 0.5D;
		pop = evaluate(pop, evaluation_, evaluations_limit_);
		Population newPop = Selection.uniformParent(pop, 11, rnd_);
		if ((Function==1)||(Function==3))
			newPop.setIndividual(0, pop.getFittest(0));
		boolean improve = false;
		do {
			if ((improve)&&(Function!=1)&&(Function!=3)) {
				for (int i = 0; i < 11; i++) {
					pop.setIndividual(POP_SIZE - 1 - i, newPop.getIndividual(i));
				}
				newPop = Selection.uniformParent(pop, 11, rnd_);
				improve = false;
			}
			//1.  Order according to the values at the vertices
			newPop.sort(Population.FITNESS);
			//2. Calculate the centroid of all points except the worst
			double[] x0 = { 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D };
			for (int i = 0; i < newPop.size() - 1; i++) {
				for (int j = 0; j < 10; j++) {
					x0[j] += newPop.getIndividual(i).getValue(j) / 10.0D;
				}
			}
			//3. reflection
			double[] reflected = new double[10];
			boolean oob = false;
			for (int j = 0; j < 10; j++) {
				reflected[j] = x0[j] + alpha * (x0[j] - newPop.getIndividual(10).getValue(j));
				if ((reflected[j] > 5.0D) || (reflected[j] < -5.0D)) oob = true;
			}
			if (evals < evaluations_limit_) {
				double fitnessReflected = -9999.0D;
				if (!oob) {
					fitnessReflected = ((Double)evaluation_.evaluate(reflected)).doubleValue();
					evals++;
				}
				//If the reflected point is better than the second worst, but not better than the best
				if ((!oob) && (fitnessReflected > newPop.getIndividual(9).getFitness()) && (fitnessReflected <= newPop.getIndividual(0).getFitness())) {
					newPop.getIndividual(10).setValues(reflected);
					newPop.getIndividual(10).setFitness(fitnessReflected);
					improve = true;
				} else if ((!oob) && (fitnessReflected > newPop.getIndividual(0).getFitness())) {
					//4. Expansion
					double[] expansion = new double[10];
					for (int j = 0; j < 10; j++) {
						expansion[j] = x0[j] + gamma * (reflected[j] - x0[j]);
						if ((expansion[j]>5)||(expansion[j]<-5)) oob=true;//added 18-10
					}
					if (evals < evaluations_limit_) {
						double fitnessExpansion = -9999;//added 18-10
						if (!oob){//added 18-10
							fitnessExpansion = ((Double)evaluation_.evaluate(expansion)).doubleValue();
							evals++;
						}//added 18-10
						//If the expanded point is better than the reflected point
						if (!oob&&(fitnessExpansion>fitnessReflected)){//added 18-10
							//if (fitnessExpansion > fitnessReflected) {
							newPop.getIndividual(10).setValues(expansion);
							newPop.getIndividual(10).setFitness(fitnessExpansion);
						} else {
							newPop.getIndividual(10).setValues(reflected);
							newPop.getIndividual(10).setFitness(fitnessReflected);
						}
						improve = true;
					}
				} else {
					//5. Contraction
					//Compute contracted point
					double[] contracted = new double[10];
					oob = false;
					for (int j = 0; j < 10; j++) {
						contracted[j] = x0[j] + rho * (newPop.getIndividual(10).getValue(j) - x0[j]);
						if ((contracted[j] > 5.0D) || (contracted[j] < -5.0D)) oob = true;
					}
					if (evals < evaluations_limit_) {
						double fitnessContraction = -9999.0D;
						if (!oob) {
							fitnessContraction = ((Double)evaluation_.evaluate(contracted)).doubleValue();
							evals++;
						}
						//If the contracted point is better than the worst point
						if ((!oob) && (fitnessContraction > newPop.getIndividual(10).getFitness())) {
							newPop.getIndividual(10).setValues(contracted);
							newPop.getIndividual(10).setFitness(fitnessContraction);
							improve = true;
						} else {
							//6. Shrink
							for (int i = 1; i < 11; i++) {
								//For all but the best point, replace the point
								for (int j = 0; j < 10; j++) {
									newPop.getIndividual(i).setValue(j, newPop.getIndividual(0).getValue(j) + sigma * (newPop.getIndividual(i).getValue(j) - newPop.getIndividual(0).getValue(j)));
								}
								if (evals < evaluations_limit_) {
									double fitness = ((Double)evaluation_.evaluate(newPop.getIndividual(i).getValues())).doubleValue();
									newPop.getIndividual(i).setFitness(fitness);
									evals++;
								}
							}
						}
					}
				}
			}
		} while (evals < evaluations_limit_);
		return pop;
	}
}

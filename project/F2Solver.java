import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import org.vu.contest.ContestEvaluation;

public class F2Solver
{
	static int SEED = 35;
	private int evals = 0;
	static int POP_SIZE = 100;
	//Memetic
	static int budget = 1000;
	private double func_factor = 0.6D;
	static double step = 0.03D;
	//DE
	static double CR = 0.2D;
	static double F = 0.5D;

	public F2Solver() {}

	public void run(player2 player, ContestEvaluation evaluation_, int evaluations_limit_, Random rnd_)
	{
		rnd_.setSeed(SEED);
		F2Population pop = new F2Population(POP_SIZE);
		pop = pop.init(rnd_);

		double totalFitness_NM = -999999.0D;
		while (evals<evaluations_limit_){
			double startFitness = totalFitness_NM;
			pop = DE(player, pop, evaluation_, Math.min((int) (evals+budget*func_factor),evaluations_limit_), CR, rnd_, false);
			double totalFitness_DE = 0.0D;
			for (int i = 0; i < POP_SIZE; i++) {
				totalFitness_DE += pop.getFittest(i).getFitness();
			}
			pop = NelderMead(player, pop, evaluation_, Math.min((int) (evals+budget*(1-func_factor)),evaluations_limit_), rnd_);
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

	public F2Population evaluate(player2 player, F2Population pop, ContestEvaluation evaluation_, int evaluations_limit_) {
		for (int index = 0; index < pop.size(); index++) {
			if ((pop.getF2Individual(index).getFitness() == -1.0D) && (evals < evaluations_limit_)) {
				double fitness = ((Double)evaluation_.evaluate(pop.getF2Individual(index).getValues())).doubleValue();
				pop.getF2Individual(index).setFitness(fitness);
				evals++;
			}
		}
		return pop;
	}

	public F2Population DE(player2 player, F2Population pop, ContestEvaluation evaluation_, int evaluations_limit_, double crossoverRate, Random rnd_, boolean RandSelect) {
		pop = evaluate(player, pop, evaluation_, evaluations_limit_);
		double F = F2Solver.F;
		F2Population newPop = new F2Population(pop.size());
		while (evals < evaluations_limit_) {
			for (int i = 0; i < pop.size(); i++) {
				if (evals < evaluations_limit_) {
					F2Individual parent = pop.getF2Individual(i);
					F2Individual offspring = new F2Individual(rnd_);
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
					F2Individual base = pop.getF2Individual(index1);
					F2Individual individual1 = pop.getF2Individual(index2);
					F2Individual individual2 = pop.getF2Individual(index3);
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
						newPop.setF2Individual(i, offspring);
					} else{
						newPop.setF2Individual(i, pop.getF2Individual(i));
					}
				}
			}
			pop = newPop;
		}
		return newPop;
	}

	public F2Population NelderMead(player2 player, F2Population pop, ContestEvaluation evaluation_, int evaluations_limit_, Random rnd_) {
		double alpha = 1.0D;
		double gamma = 2.0D;
		double rho = 0.5D;
		double sigma = 0.5D;
		pop = evaluate(player, pop, evaluation_, evaluations_limit_);
		F2Population newPop = uniformParent(pop, 11, rnd_);
		boolean improve = false;
		do {
			if (improve) {
				for (int i = 0; i < 11; i++) {
					pop.setF2Individual(POP_SIZE - 1 - i, newPop.getF2Individual(i));
				}
				newPop = uniformParent(pop, 11, rnd_);
				improve = false;
			}
			//1.  Order according to the values at the vertices
			newPop.sort(F2Population.FITNESS);
			//2. Calculate the centroid of all points except the worst
			double[] x0 = { 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D, 0.0D };
			for (int i = 0; i < newPop.size() - 1; i++) {
				for (int j = 0; j < 10; j++) {
					x0[j] += newPop.getF2Individual(i).getValue(j) / 10.0D;
				}
			}
			//3. reflection
			double[] reflected = new double[10];
			boolean oob = false;
			for (int j = 0; j < 10; j++) {
				reflected[j] = x0[j] + alpha * (x0[j] - newPop.getF2Individual(10).getValue(j));
				if ((reflected[j] > 5.0D) || (reflected[j] < -5.0D)) oob = true;
			}
			if (evals < evaluations_limit_) {
				double fitnessReflected = -9999.0D;
				if (!oob) {
					fitnessReflected = ((Double)evaluation_.evaluate(reflected)).doubleValue();
					evals++;
				}
				//If the reflected point is better than the second worst, but not better than the best
				if ((!oob) && (fitnessReflected > newPop.getF2Individual(9).getFitness()) && (fitnessReflected <= newPop.getF2Individual(0).getFitness())) {
					newPop.getF2Individual(10).setValues(reflected);
					newPop.getF2Individual(10).setFitness(fitnessReflected);
					improve = true;
				} else if ((!oob) && (fitnessReflected > newPop.getF2Individual(0).getFitness())) {
					//4. Expansion
					double[] expansion = new double[10];
					for (int j = 0; j < 10; j++) {
						expansion[j] = x0[j] + gamma * (reflected[j] - x0[j]);
					}
					if (evals < evaluations_limit_) {
						double fitnessExpansion = ((Double)evaluation_.evaluate(expansion)).doubleValue();
						evals++;
						//If the expanded point is better than the reflected point
						if (fitnessExpansion > fitnessReflected) {
							newPop.getF2Individual(10).setValues(expansion);
							newPop.getF2Individual(10).setFitness(fitnessExpansion);
						} else {
							newPop.getF2Individual(10).setValues(reflected);
							newPop.getF2Individual(10).setFitness(fitnessReflected);
						}
						improve = true;
					}
				} else {
					//5. Contraction
					//Compute contracted point
					double[] contracted = new double[10];
					oob = false;
					for (int j = 0; j < 10; j++) {
						contracted[j] = x0[j] + rho * (newPop.getF2Individual(10).getValue(j) - x0[j]);
						if ((contracted[j] > 5.0D) || (contracted[j] < -5.0D)) oob = true;
					}
					if (evals < evaluations_limit_) {
						double fitnessContraction = -9999.0D;
						if (!oob) {
							fitnessContraction = ((Double)evaluation_.evaluate(contracted)).doubleValue();
							evals++;
						}
						//If the contracted point is better than the worst point
						if ((!oob) && (fitnessContraction > newPop.getF2Individual(10).getFitness())) {
							newPop.getF2Individual(10).setValues(contracted);
							newPop.getF2Individual(10).setFitness(fitnessContraction);
							improve = true;
						}
						else {
							//6. Shrink
							for (int i = 1; i < 11; i++) {
								//For all but the best point, replace the point
								for (int j = 0; j < 10; j++) {
									newPop.getF2Individual(i).setValue(j, newPop.getF2Individual(0).getValue(j) + sigma * (newPop.getF2Individual(i).getValue(j) - newPop.getF2Individual(0).getValue(j)));
								}
								if (evals < evaluations_limit_) {
									double fitness = ((Double)evaluation_.evaluate(newPop.getF2Individual(i).getValues())).doubleValue();
									newPop.getF2Individual(i).setFitness(fitness);
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

	public F2Population uniformParent(F2Population parentPop, int selectionSize, Random rnd_)
	{
		F2Population selectedPop = new F2Population(selectionSize);
		for (int i = 0; i < selectionSize; i++) {
			int index = rnd_.nextInt(parentPop.size() - i);
			F2Individual selectedF2Individual = parentPop.getF2Individual(index);
			selectedPop.setF2Individual(i, selectedF2Individual.clone());
			parentPop.setF2Individual(index, parentPop.getF2Individual(parentPop.size() - 1 - i));
			parentPop.setF2Individual(parentPop.size() - 1 - i, selectedF2Individual);
		}
		return selectedPop;
	}


	public class F2Population {
		static final int FITNESS = 3;
		private F2Individual population[];

		public F2Population(int populationSize) { 
			this.population = new F2Individual[populationSize]; 
		} 
		public F2Population init(Random rnd_) {
			for (int i = 0; i < this.size(); i++) {
				this.population[i] = new F2Individual(rnd_);
			}
			return this;
		}
		public F2Population sort(int variable) {
			Arrays.sort(this.population, new Comparator<F2Individual>() {
				@Override
				public int compare(F2Individual o1, F2Individual o2) {
					if (o1.getFitness() > o2.getFitness()) {
						return -1;
					} else if (o1.getFitness() < o2.getFitness()) {
						return 1;
					} else return 0;
				}
			});
			return this;
		}
		public F2Individual getFittest(int index) {
			this.sort(FITNESS);
			return this.population[index];
		}
		public int size() {
			return this.population.length;
		}
		public F2Individual setF2Individual(int index, F2Individual individual) {
			return population[index] = individual;
		}
		public F2Individual getF2Individual(int index) {
			return population[index];
		}
	}

	public class F2Individual {
		private double[] values;
		private double fitness = -1;

		public F2Individual(){
		}

		public F2Individual(Random rnd_) {
			double[] values = new double[10];
			for (int i = 0; i < 10; i++) {
				double rnd = rnd_.nextDouble()*10-5;
				values[i] = rnd;
			}
			this.values = values;
		}
		public F2Individual clone() {
			F2Individual clone = new F2Individual();
			clone.values = this.values.clone();
			clone.fitness = this.fitness;
			return clone;
		}
		public double[] getValues() {
			return this.values;
		}
		public void setValues(double[] values) {
			this.values = values;
		}
		public void setValue(int index, double value) {
			this.values[index] = value;
		}
		public double getValue(int index) {
			return this.values[index];
		}
		public void setFitness(double fitness) {
			this.fitness = fitness;
		}
		public double getFitness() {
			return this.fitness;
		}
	}
}
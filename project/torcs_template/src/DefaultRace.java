import cicontest.algorithm.abstracts.AbstractRace;
import cicontest.algorithm.abstracts.DriversUtils;
import cicontest.algorithm.abstracts.map.TrackMap;
import cicontest.torcs.controller.Driver;
import cicontest.torcs.controller.Human;
import cicontest.torcs.race.Race;
import cicontest.torcs.race.RaceResult;
import cicontest.torcs.race.RaceResults;
import scr.Controller;

public class DefaultRace extends AbstractRace {

	public double[] runQualification2(Driver[] drivers, boolean withGUI) {
		double[] fitness = new double[drivers.length];
		Race race = new Race();
		race.setTrack(this.tracktype, this.track);
		race.setTermination(Race.Termination.LAPS, this.laps);
		race.setStage(Controller.Stage.QUALIFYING);
		Driver[] results = drivers;
		int i = drivers.length;

		for(int var7 = 0; var7 < i; ++var7) {
			Driver driver = results[var7];
			race.addCompetitor(driver);
		}

		RaceResults var9;
		if(withGUI) {
			var9 = race.runWithGUI();
		} else {
			var9 = race.run();
		}

		for(i = 0; i < drivers.length; ++i) {
			fitness[i] = ((RaceResult)var9.get(drivers[i])).getBestLapTime();
		}

		this.printResults(drivers, var9);
		return fitness;
	}

	public int[] runQualification(DefaultDriverGenome[] drivers, boolean withGUI){
		DefaultDriver[] driversList = new DefaultDriver[drivers.length + 1 ];
		for(int i=0; i<drivers.length; i++){
			driversList[i] = new DefaultDriver();
			driversList[i].loadGenome(drivers[i]);
		}
		return runQualification(driversList, withGUI);
	}

	
	public int[] runRace(DefaultDriverGenome[] drivers, boolean withGUI){
		int size = Math.min(10, drivers.length);
		DefaultDriver[] driversList = new DefaultDriver[size];
		for(int i=0; i<size; i++){
			driversList[i] = new DefaultDriver();
			driversList[i].loadGenome(drivers[i]);
		}
		return runRace(driversList, withGUI, true);
	}

	
	
	public void showBest(){
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
		
		DefaultDriverGenome best = (DefaultDriverGenome) DriversUtils.getStoredGenome();
		DefaultDriver driver = new DefaultDriver();
		driver.loadGenome(best);
		
		DefaultDriver[] driversList = new DefaultDriver[1];
		driversList[0] = driver;
		runQualification(driversList, true);
	}
	
	public void showBestRace(){
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
	
		DefaultDriver[] driversList = new DefaultDriver[1];
		
		for(int i=0; i<10; i++){
			DefaultDriverGenome best = (DefaultDriverGenome) DriversUtils.getStoredGenome();
			DefaultDriver driver = new DefaultDriver();
			driver.loadGenome(best);
			driversList[i] = driver;
		}
		
		runRace(driversList, true, true);
	}
	
	public void raceBest(){
		
		if(DriversUtils.getStoredGenome() == null ){
			System.err.println("No best-genome known");
			return;
		}
		
		Driver[] driversList = new Driver[10];
		for(int i=0; i<10; i++){
			DefaultDriverGenome best = (DefaultDriverGenome) DriversUtils.getStoredGenome();
			DefaultDriver driver = new DefaultDriver();
			driver.loadGenome(best);
			driversList[i] = driver;
		}
		driversList[0] = new Human();
		runRace(driversList, true, true);
	}
}

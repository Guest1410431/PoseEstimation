package main;

import java.awt.EventQueue;

import pose.encoder.PoseNet;
import pose.trainer.DataLoader;
import pose.trainer.Dataset;
import pose.trainer.MSELoss;
import pose.trainer.SGDOptimizer;
import pose.trainer.Trainer;
import window.Window;

public class Main
{
	public static void main(String[] args)
	{
		PoseNet poseNet = new PoseNet();
		
		Dataset dataset = DataLoader.loadDataset("res/training_data/tensorImages", "res/training_data/heatmaps");
		
		Trainer trainer = new Trainer(poseNet, dataset, new SGDOptimizer(0.001f), new MSELoss());
		
		trainer.train(1);
		/*
		 EventQueue.invokeLater(new Runnable()
		 
		{
			public void run()
			{
				try
				{
					// Window window = new Window();
					// window.frame.setVisible(true);
				}
				catch (Exception e)
				{
					e.printStackTrace();
				}
			}
		});
		*/
	}
}

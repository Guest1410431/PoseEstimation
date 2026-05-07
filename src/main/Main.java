package main;

import java.awt.EventQueue;

import preprocess.ImagePreprocessing;
import window.Window;

public class Main
{
	public static void main(String[] args)
	{
		ImagePreprocessing imagePreprocess = new ImagePreprocessing();

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
	}
}

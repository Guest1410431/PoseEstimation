package main;

import java.awt.EventQueue;

import window.Window;

public class Main
{
	public static void main(String[] args)
	{
		
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

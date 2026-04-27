package window;

import javax.swing.JFrame;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.LayoutStyle.ComponentPlacement;

import javax.swing.JSlider;

public class Window
{
	public JFrame frame;

	private VideoController videoController;
	private VideoPanel videoPanel;
	private JButton btnLoad;
	private JButton btnPlay;
	static JSlider videoProgressSlider;
	
	public Window()
	{
		videoController = new VideoController();
		initialize();
	}

	private void initialize()
	{
		frame = new JFrame();
		frame.setBounds(0, 0, 1400, 900);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		videoPanel = new VideoPanel(videoController);
		videoPanel.setBackground(new Color(255, 255, 255));

		btnLoad = new JButton("Load");
		btnPlay = new JButton(">");
		btnPlay.setEnabled(false);
		videoProgressSlider = new JSlider(0);
		videoProgressSlider.setSnapToTicks(false);

		btnLoad.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				videoPanel.BufferReady();

				videoProgressSlider.setMaximum(videoPanel.getNumFrames());
				btnPlay.setEnabled(true);
			}
		});
		btnPlay.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e)
			{
				if(videoController.playing)
				{
					btnPlay.setText(">");
					videoController.playing = false;
				}
				else
				{
					btnPlay.setText("||");
					videoController.playing = true;
				}
			}
		});
		videoProgressSlider.addChangeListener(e -> {
			if (videoController.updatingUI)
			{
				return;
			}
			videoController.latestSeekFrame.set(videoProgressSlider.getValue());
			videoController.playing = false;
			btnPlay.setText(">");
		});

		GroupLayout groupLayout = new GroupLayout(frame.getContentPane());
		groupLayout.setHorizontalGroup(groupLayout.createParallelGroup(Alignment.LEADING).addGroup(groupLayout
				.createSequentialGroup().addContainerGap()
				.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(videoPanel, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
						.addGroup(groupLayout.createSequentialGroup()
								.addComponent(btnLoad, GroupLayout.PREFERRED_SIZE, 95, GroupLayout.PREFERRED_SIZE)
								.addGap(2)
								.addComponent(btnPlay, GroupLayout.PREFERRED_SIZE, 80, GroupLayout.PREFERRED_SIZE)
								.addGap(49)
								.addComponent(videoProgressSlider, GroupLayout.DEFAULT_SIZE, 1150, Short.MAX_VALUE)))
				.addContainerGap()));
		groupLayout.setVerticalGroup(groupLayout.createParallelGroup(Alignment.TRAILING).addGroup(groupLayout
				.createSequentialGroup().addComponent(videoPanel, GroupLayout.DEFAULT_SIZE, 810, Short.MAX_VALUE)
				.addPreferredGap(ComponentPlacement.RELATED)
				.addGroup(groupLayout.createParallelGroup(Alignment.LEADING)
						.addComponent(btnPlay, GroupLayout.DEFAULT_SIZE, GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
						.addGroup(groupLayout.createParallelGroup(Alignment.LEADING, false)
								.addComponent(videoProgressSlider, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE,
										GroupLayout.PREFERRED_SIZE)
								.addComponent(btnLoad, GroupLayout.DEFAULT_SIZE, 40, Short.MAX_VALUE)))
				.addContainerGap()));
		frame.getContentPane().setLayout(groupLayout);
	}

	public static void setProgressBar(int frameIndex, int numFrames)
	{
		videoProgressSlider.setValue((frameIndex * 100) / numFrames);
	}
}

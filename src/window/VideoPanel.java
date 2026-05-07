package window;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class VideoPanel extends JPanel
{
	private static final long serialVersionUID = 1L;

	private VideoController controller;

	private static Thread videoThread;

	private static Mat frame;
	private BufferedImage imgBuffer;

	private int numFrames;

	private String filePath;

	private static VideoCapture video;

	public VideoPanel(VideoController controller)
	{
		this.controller = controller;
	}

	public void loadVideo(String filePath)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		this.filePath = filePath;
		video = new VideoCapture(filePath);
		frame = new Mat();
		numFrames = (int) video.get(Videoio.CAP_PROP_FRAME_COUNT);
	}

	private BufferedImage matToBufferedImage(Mat mat)
	{
		MatOfByte mob = new MatOfByte();
		Imgcodecs.imencode(".jpg", mat, mob);

		try
		{
			return ImageIO.read(new ByteArrayInputStream(mob.toArray()));
		} catch (IOException e)
		{
			e.printStackTrace();
			return null;
		}
	}

	public void BufferReady()
	{
		if (video.isOpened())
		{
			video.release();
		}
		video.open(filePath);
		frame = new Mat();

		videoThread = new Thread(movRunner);
		videoThread.setDaemon(true);
		videoThread.start();

		repaint();
	}

	private final Runnable movRunner = new Runnable()
	{
		public void run()
		{
			boolean frameRefresh = true;

			while (true)
			{
				int seekFrame = controller.latestSeekFrame.getAndSet(-1);

				if (seekFrame >= 0)
				{
					controller.frame = seekFrame;
					video.set(Videoio.CAP_PROP_POS_FRAMES, seekFrame);
					frameRefresh = true;
				}
				if (frameRefresh)
				{
					if (!video.read(frame) || frame.empty())
					{
						controller.playing = false;
						continue;
					}
					imgBuffer = matToBufferedImage(frame);
					repaint();
					frameRefresh = false;
				}
				controller.updatingUI = true;
				SwingUtilities.invokeLater(() -> {
					controller.updatingUI = true;
					Window.videoProgressSlider.setValue(controller.frame);
					controller.updatingUI = false;
				});
				if (controller.playing)
				{
					controller.frame++;
					frameRefresh = true;
				}
				sleepQuiet(15);
			}
		}
	};

	public void sleepQuiet(int sleepTime)
	{
		try
		{
			Thread.sleep(sleepTime);
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}

	public int getNumFrames()
	{
		return numFrames;
	}

	protected void paintComponent(Graphics g)
	{
		super.paintComponent(g);

		if (imgBuffer != null)
		{
			Graphics2D g2d = (Graphics2D) g.create();
			int x = (getWidth() - imgBuffer.getWidth()) / 2;
			int y = (getHeight() - imgBuffer.getHeight()) / 2;
			g2d.drawImage(imgBuffer, x, y, this);
			g2d.dispose();
		}
	}
}

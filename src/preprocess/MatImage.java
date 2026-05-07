package preprocess;

import org.opencv.core.Mat;

public class MatImage
{
	private Mat mat;
	private String imageName;

	public MatImage(Mat mat, String imageName)
	{
		this.mat = mat;
		this.imageName = imageName;
	}

	public Mat getMat()
	{
		return mat;
	}

	public String getImageName()
	{
		return imageName;
	}

	@Override
	public boolean equals(Object o)
	{
		if (this == o)
		{
			return true;
		}
		if (!(o instanceof MatImage))
		{
			return false;
		}
		MatImage other = (MatImage) o;
		return imageName.equals(other.imageName);
	}

	@Override
	public int hashCode()
	{
		return imageName.hashCode();
	}
}

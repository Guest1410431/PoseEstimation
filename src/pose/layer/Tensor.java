package pose.layer;

public class Tensor
{
	private final float[] data;
	private final int[] shape;

	public Tensor(int... shape)
	{
		this.shape = shape;
		int size = 1;

		for (int dim : shape)
		{
			size *= dim;
		}
		this.data = new float[size];
	}

	public Tensor(float[] data, int... shape)
	{
		this.data = data;
		this.shape = shape;
	}

	public float[] getData()
	{
		return data;
	}

	public int[] getShape()
	{
		return shape;
	}
}

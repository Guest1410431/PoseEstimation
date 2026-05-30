package pose.layer;

public class DownSampleLayer extends Layer
{
	private final int kernalSize;
	private final int stride;

	public DownSampleLayer(int kernalSize, int stride)
	{
		super();
		this.kernalSize = kernalSize;
		this.stride = stride;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// TODO: Pooling
		int batchSize = input.getBatchSize();
		int channels = input.getChannels();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = (inHeight - kernalSize) / stride + 1;
		int outWidth = (inWidth - kernalSize) / stride + 1;

		Tensor output = new Tensor(batchSize, channels, outHeight, outWidth);

		float[] inData = input.getData();
		float[] outData = output.getData();

		for(int batch = 0; batch < batchSize; batch++)
		{
			for(int channel = 0; channel < channels; channel++)
			{
				for(int outY = 0; outY < outHeight; outY++)
				{
					for(int outX = 0; outX < outWidth; outX++)
					{
						float maxVal = -Float.MAX_VALUE;
						
						for(int y = 0; y < kernalSize; y++)
						{
							for(int x = 0; x < kernalSize; x++)
							{
								int inY = outY * stride + y;
								int inX = outX * stride + x;
								
								float value = input.get(batch, channel, inY, inX);
								
								if(value > maxVal)
								{
									maxVal = value;
								}
							}
							output.set(batch, channel, outY, outX, maxVal);
						}
					}
				}
			}
		}
		return output;
	}
}












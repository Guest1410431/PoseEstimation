package pose.layer;

public class UpSampleLayer extends Layer
{
	private final int scale;

	public UpSampleLayer(int scale)
	{
		this.scale = scale;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// Nearest Neighbors
		this.input = input;

		int batchSize = input.getBatchSize();
		int channels = input.getChannels();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = inHeight * scale;
		int outWidth = inWidth * scale;

		Tensor output = new Tensor(batchSize, channels, outHeight, outWidth);

		float[] in = input.getData();
		float[] out = output.getData();

		int inHeightWidth = inHeight * inWidth;
		int outHeightWidth = outHeight * outWidth;

		for (int batch = 0; batch < batchSize; batch++)
		{
			int inBatch = batch * channels * inHeightWidth;
			int outBatch = batch * channels * outHeightWidth;

			for (int channel = 0; channel < channels; channel++)
			{
				int inChannel = inBatch + channel * inHeightWidth;
				int outChannel = outBatch + channel * outHeightWidth;
				
				int inIndex = inChannel;
				
				for (int y = 0; y < inHeight; y++)
				{
					int outY = y * scale;

					for (int x = 0; x < inWidth; x++, inIndex++)
					{
						float value = in[inIndex];

						int outX = x * scale;

						for (int dy = 0; dy < scale; dy++)
						{
							int row = outChannel + (outY + dy) * outWidth + outX;
							int outIndex = row;
							
							for (int dx = 0; dx < scale; dx++, outIndex++)
							{
								out[outIndex] = value;
							}
						}
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		int batchSize = gradient.getBatchSize();
		int channels = gradient.getChannels();
		int outHeight = gradient.getHeight() / scale;
		int outWidth = gradient.getWidth() / scale;
		int gradientWidth = gradient.getWidth();

		Tensor gradOutput = new Tensor(batchSize, channels, outHeight, outWidth);

		float[] grad = gradient.getData();
		float[] out = gradOutput.getData();

		int gradHeightWidth = gradient.getHeight() * gradient.getWidth();
		int outHeightWidth = outHeight * outWidth;

		for (int batch = 0; batch < batchSize; batch++)
		{
			int gradBatch = batch * channels * gradHeightWidth;
			int outBatch = batch * channels * outHeightWidth;

			for (int channel = 0; channel < channels; channel++)
			{
				int gradChannel = gradBatch + channel * gradHeightWidth;
				int outChannel = outBatch + channel * outHeightWidth;

				int outIndex = outChannel;
				
				for (int y = 0; y < outHeight; y++)
				{
					int startY = y * scale;

					for (int x = 0; x < outWidth; x++, outIndex++)
					{
						int startX = x * scale;

						float sum = 0f;

						for (int dy = 0; dy < scale; dy++)
						{
							int row = gradChannel + (startY + dy) * gradientWidth + startX;
							int gradIndex = row;
							
							for (int dx = 0; dx < scale; dx++, gradIndex++)
							{
								sum += grad[gradIndex];
							}
						}
						out[outIndex] = sum;
					}
				}
			}
		}
		return gradOutput;
	}
}

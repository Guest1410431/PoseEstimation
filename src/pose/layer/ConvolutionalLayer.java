package pose.layer;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ConvolutionalLayer extends Layer
{
	private final int inChannels;
	private final int outChannels;
	private final int kernalSize;
	private final int stride;
	private final int padding;
	
	private final float[] weights;
	private final float[] bias;

	public ConvolutionalLayer(int inChannels, int outChannels, int kernalSize, int stride, int padding)
	{
		this.inChannels = inChannels;
		this.outChannels = outChannels;
		this.kernalSize = kernalSize;
		this.stride = stride;
		this.padding = padding;

		int weightCount = outChannels * inChannels * kernalSize * kernalSize;
		this.weights = new float[weightCount];
		this.bias = new float[outChannels];

		// Uniform Weight Distribution
		int fanIn = inChannels * kernalSize * kernalSize;
		double a = Math.sqrt(6 / fanIn);
		Mat w = new Mat(outChannels, fanIn, CvType.CV_32F);
		Core.randu(w, -a, a);
	}

	@Override
	public Tensor forward(Tensor input)
	{
		int batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = Math.floorDiv(inHeight + (padding * 2) - kernalSize, stride) + 1;
		int outWidth = Math.floorDiv(inWidth + (padding * 2) - kernalSize, stride) + 1;

		Tensor output = new Tensor(batchSize, outWidth, outHeight, outChannels );

		for(int batch =0; batch < batchSize; batch++)
		{
			for (int chout = 0; chout < outChannels; chout++)
			{
				for (int outH = 0; outH < outHeight; outH++)
				{
					for (int outW = 0; outW < outWidth; outW++)
					{
						float sum = bias[chout];

						for (int chin = 0; chin < inChannels; chin++)
						{
							for (int kernalX = 0; kernalX < kernalSize; kernalX++)
							{
								for (int kernalY = 0; kernalY < kernalSize; kernalY++)
								{
									int inX = outW * stride + kernalX - padding;
									int inY = outH * stride + kernalY - padding;

									if (inY >= 0 && inX >= 0 && inY < inHeight && inX < inWidth)
									{
										int dataIndex = inY * (inWidth * inChannels) + inX * inChannels + chin;
										int weightIndex = chout * (kernalSize * kernalSize * inChannels) + kernalY * (kernalSize * inChannels) + kernalX * inChannels + chin;

										sum += input.getData()[dataIndex] * weights[weightIndex];
									}
								}
							}
						}
						output.set(batch, outH, outW, chout, sum);
					}
				}
			}
		}
		return output;
	}
}

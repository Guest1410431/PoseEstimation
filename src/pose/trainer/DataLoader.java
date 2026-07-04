package pose.trainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import java.io.File;

import pose.tensor.Tensor;
import pose.tensor.TensorReader;

public class DataLoader
{
	private final Dataset dataset;
	private final int batchSize;
	private int currentIndex;

	public DataLoader(Dataset dataset, int batchSize)
	{
		this.dataset = dataset;
		this.batchSize = batchSize;
		currentIndex = 0;
	}

	public boolean hasNext()
	{
		return currentIndex < dataset.size();
	}

	public Batch next()
	{
		int actualBatchSize = Math.min(batchSize, dataset.size() - currentIndex);

		Tensor image = TensorReader.loadTensor(dataset.get(currentIndex).getImageTensorPath());
		Tensor heatmap = TensorReader.loadTensor(dataset.get(currentIndex).getHeatmapTensorPath());

		int imageChannels = image.getChannels();
		int imageHeight = image.getHeight();
		int imageWidth = image.getWidth();

		int heatmapChannels = heatmap.getChannels();
		int heatmapHeight = heatmap.getHeight();
		int heatmapWidth = heatmap.getWidth();

		Tensor imageBatch = Tensor.acquire(actualBatchSize, imageChannels, imageHeight, imageWidth);
		Tensor heatmapBatch = Tensor.acquire(actualBatchSize, heatmapChannels, heatmapHeight, heatmapWidth);

		copyIntoBatch(image, imageBatch, 0);
		copyIntoBatch(heatmap, heatmapBatch, 0);

		Tensor.release(image);
		Tensor.release(heatmap);

		for (int i = 0; i < actualBatchSize; i++)
		{
			DataInfo tensorInfo = dataset.get(currentIndex + i);

			Tensor imageTensor = TensorReader.loadTensor(tensorInfo.getImageTensorPath());
			Tensor heatmapTensor = TensorReader.loadTensor(tensorInfo.getHeatmapTensorPath());

			copyIntoBatch(imageTensor, imageBatch, i);
			copyIntoBatch(heatmapTensor, heatmapBatch, i);

			Tensor.release(imageTensor);
			Tensor.release(heatmapTensor);
		}
		currentIndex += actualBatchSize;

		return new Batch(imageBatch, heatmapBatch);
	}

	private void copyIntoBatch(Tensor source, Tensor destination, int index)
	{
		int channels = source.getChannels();
		int height = source.getHeight();
		int width = source.getWidth();

		for (int c = 0; c < channels; c++)
		{
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					destination.set(index, c, y, x, source.get(0, c, y, x));
				}
			}
		}
	}

	public static Dataset loadDataset(String imageTensorPath, String heatmapTensorPath)
	{
		List<DataInfo> dataInfo = new ArrayList<DataInfo>();

		File[] imageFiles = new File(imageTensorPath).listFiles((dir, name) -> name.endsWith(".tensor"));

		if (imageFiles == null)
		{
			throw new IllegalStateException("No image tensors found");
		}
		Arrays.sort(imageFiles);

		for (File imageFile : imageFiles)
		{
			File heatmapFile = new File(heatmapTensorPath, imageFile.getName());

			if (!heatmapFile.exists())
			{
				throw new RuntimeException("No heatmap tensor found for " + imageFile.getName());
			}
			dataInfo.add(new DataInfo(imageFile.getAbsolutePath(), heatmapFile.getAbsolutePath()));
		}
		return new Dataset(dataInfo);
	}

	public void shuffle()
	{
		dataset.shuffle();
	}

	public void reset()
	{
		currentIndex = 0;
	}
}

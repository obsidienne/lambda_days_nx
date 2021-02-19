defmodule LambdaDaysNx do
  @moduledoc """
  Documentation for `LambdaDaysNx`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> LambdaDaysNx.hello()
      :world

  """
  def hello do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      File.read!("train-images-idx3-ubyte.gz") |> :zlib.gunzip()

    heatmap =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows, n_cols})
      |> Nx.to_heatmap()

    images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
      |> Nx.divide(255)
      |> Nx.to_batched_list(30)

    <<_::32, n_labels::32, labels::binary>> =
      File.read!("train-labels-idx1-ubyte.gz") |> :zlib.gunzip()

    labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched_list(30)
  end
end

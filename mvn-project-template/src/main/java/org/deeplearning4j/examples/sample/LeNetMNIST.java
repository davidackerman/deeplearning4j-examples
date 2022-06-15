///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.deeplearning4j:deeplearning4j-core:1.0.0-M2
//DEPS org.nd4j:nd4j-native:1.0.0-M2
/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.examples.sample;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import bdv.util.AxisOrder;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvHandle;
import bdv.util.BdvSource;
import de.embl.cba.bdv.utils.BdvUtils;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

/**
 * Created by agibsonccc on 9/16/15.
 */
public class LeNetMNIST {
    public static boolean draw = false;
    
    static class KeyListenerDemo extends KeyAdapter {
	@Override
	public void keyPressed(KeyEvent event) {

	    char ch = event.getKeyChar();

	    if (ch == 'd' ) {
		draw = true;
	    }

	    
	}
	public void keyReleased(KeyEvent event) {

	    char ch = event.getKeyChar();

	    if (ch == 'd' ) {
		draw = false;
	    }

	    
	}
	
    }
    
    static class MouseMotionEventDemo implements MouseMotionListener {
	RandomAccess<FloatType> bdvRA = null;
	BdvHandle bdvHandle = null;
	MultiLayerNetwork model = null;
	public MouseMotionEventDemo(BdvHandle bdvHandle, RandomAccess<FloatType> bdvRA, MultiLayerNetwork model) {
	    this.bdvHandle = bdvHandle;
	    this.bdvRA = bdvRA;
	    this.model = model;
	}
	public void test() {
	    System.out.println("hasdkasdkjahsdf");
	}

	public void mouseMoved(MouseEvent e) {
	    if (draw) {
		final RealPoint position = BdvUtils.getGlobalMouseCoordinates(bdvHandle);
		final int timepoint = bdvHandle.getViewerPanel().state().getCurrentTimepoint();
		//System.out.println(position);
		this.bdvRA.setPosition(new int[] {(int) position.getDoublePosition(0), (int)position.getDoublePosition(1), timepoint});
		this.bdvRA.get().set(this.bdvRA.get().get()+(float)0.2);
		this.bdvHandle.getViewerPanel().requestRepaint();
		INDArray input = Nd4j.zeros(1, 784);
		int loc = 0;
		for(int x=0; x<28; x++) {
		    for(int y=0; y<28; y++) {
			bdvRA.setPosition(new int[] {y, x, timepoint});
			input.putScalar(loc, bdvRA.get().get());
			//input.putScalar(loc, bdvRA.get().get()); 
			loc++;
		    }
		}
		float[] output = model.output(input).toFloatVector();
		float max=-1;
		int label=-1;
		for(int i=0; i<10; i++) {
		    if(output[i]>max) {
			max = output[i];
			label = i;
		    }
		}

		bdvHandle.getViewerPanel().showMessage( label + ", " + max );
		//saySomething("Mouse moved", e);
	    }

	}

	public void mouseDragged(MouseEvent e) {
	    saySomething("Mouse dragged", e);
	}

	void saySomething(String eventDescription, MouseEvent e) {
	    System.out.println(eventDescription 
		    + " (" + e.getX() + "," + e.getY() + ")"
		    + " detected on "
		    + e.getComponent().getClass().getName());
	}
    }

    private static final Logger log = LoggerFactory.getLogger(LeNetMNIST.class);

    public static void main(String[] args) throws Exception {
	int nChannels = 1; // Number of input channels
	int outputNum = 10; // The number of possible outcomes
	int batchSize = 64; // Test batch size
	int nEpochs = 1; // Number of training epochs
	int seed = 123; //

	/*
            Create an iterator using the batch size for one iteration
	 */
	log.info("Load data....");
	DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
	DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);
	DataSet ds = mnistTrain.next();
	//System.out.println(Arrays.toString(mnistTrain.next().getFeatures().getRow(0,true).shape()));

	int[] imgDim = new int[] {28,28};
	long[] bdvDim = new long[] {28, 28, 10};
	Img<FloatType> bdvImg = ArrayImgs.floats(bdvDim);
	for (int i=0; i<10; i++) {
	    INDArray img = ds.getFeatures().getRow(i).reshape(imgDim);
	    //System.out.println(img.dataType());
	    RandomAccess<FloatType> bdvImgRA = bdvImg.randomAccess();
	    for(int x=0; x<img.shape()[0]; x++) {
		for(int y=0; y<img.shape()[1]; y++) {
		    bdvImgRA.setPosition(new int[] {x, y, i} );
		    bdvImgRA.get().set(img.getFloat(y,x));
		}
	    }
	}

	BdvSource bdv = BdvFunctions.show(bdvImg, "img", Bdv.options().is2D().axisOrder(AxisOrder.XYT));
	bdv.setDisplayRange(0,1);
	BdvHandle bdvHandle = bdv.getBdvHandle();
	String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");
	MultiLayerNetwork model = MultiLayerNetwork.load(new File(path), true);
	bdvHandle.getViewerPanel().getDisplay().addMouseMotionListener(new MouseMotionEventDemo(bdvHandle, bdvImg.randomAccess(), model));
	bdvHandle.getViewerPanel().getDisplay().addKeyListener(new KeyListenerDemo());
	//final RealPoint globalMouseCoordinates = BdvUtils.getGlobalMouseCoordinates( bdvHandle );
	//final int timepoint = bdvHandle.getViewerPanel().state().getCurrentTimepoint();

	/*
            Construct the neural network
	 */
	//	        log.info("Build model....");
	//	
	//	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	//	                .seed(seed)
	//	                .l2(0.0005)
	//	                .weightInit(WeightInit.XAVIER)
	//	                .updater(new Adam(1e-3))
	//	                .list()
	//	                .layer(new ConvolutionLayer.Builder(5, 5)
	//	                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
	//	                        .nIn(nChannels)
	//	                        .stride(1,1)
	//	                        .nOut(20)
	//	                        .activation(Activation.IDENTITY)
	//	                        .build())
	//	                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
	//	                        .kernelSize(2,2)
	//	                        .stride(2,2)
	//	                        .build())
	//	                .layer(new ConvolutionLayer.Builder(5, 5)
	//	                        //Note that nIn need not be specified in later layers
	//	                        .stride(1,1)
	//	                        .nOut(50)
	//	                        .activation(Activation.IDENTITY)
	//	                        .build())
	//	                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
	//	                        .kernelSize(2,2)
	//	                        .stride(2,2)
	//	                        .build())
	//	                .layer(new DenseLayer.Builder().activation(Activation.RELU)
	//	                        .nOut(500).build())
	//	                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	//	                        .nOut(outputNum)
	//	                        .activation(Activation.SOFTMAX)
	//	                        .build())
	//	                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
	//	                .build();
	//	
	//	        /*
	//	        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
	//	        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
	//	            and the dense layer
	//	        (b) Does some additional configuration validation
	//	        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
	//	            layer based on the size of the previous layer (but it won't override values manually set by the user)
	//	
	//	        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
	//	        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
	//	        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
	//	        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
	//	        */
	//	
	//	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	//	        model.init();
	//	
	//	        log.info("Train model...");
	//	        model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
	//	        model.fit(mnistTrain, nEpochs);
	//	
	//	        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");
	//	
	//	        log.info("Saving model to tmp folder: "+path);
	//	        model.save(new File(path), true);
	//	        MultiLayerNetwork model = MultiLayerNetwork.load(new File(path), true);
	//	        System.out.println(model.output(mnistTrain.next().getFeatures().getRow(0,true)));
	log.info("****************Example finished********************");
    }
}

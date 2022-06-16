package org.deeplearning4j.modelimportexamples.onnx;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.bdv.N5Source;
import org.janelia.saalfeldlab.n5.bdv.N5VolatileSource;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner;

import bdv.util.AxisOrder;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvHandle;
import bdv.util.BdvOptions;
import bdv.util.BdvSource;
import bdv.util.volatiles.SharedQueue;
import de.embl.cba.bdv.utils.BdvUtils;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;

public class Interactive < T extends NumericType< T >, V extends NumericType<V> >{

    static boolean draw = false;
    static class KeyListenerDemo extends KeyAdapter {
	
	RandomAccess<UnsignedByteType> rawRA = null;
	Bdv bdv = null;
	BdvHandle bdvHandle = null;
	OnnxRuntimeRunner onnxRuntimeRunner = null;
	public KeyListenerDemo(Bdv bdv, RandomAccess<UnsignedByteType> randomAccess, OnnxRuntimeRunner onnxRuntimeRunner) {
	    this.bdv = bdv;
	    this.bdvHandle = bdv.getBdvHandle();
	    this.rawRA = randomAccess;
	    this.onnxRuntimeRunner = onnxRuntimeRunner;
	}
	@Override
	public void keyPressed(KeyEvent event) {
	    

	    char ch = event.getKeyChar();

	    if (ch == 'd' ) {
		draw = true;
		final RealPoint position = BdvUtils.getGlobalMouseCoordinates(bdvHandle);
		//final int timepoint = bdvHandle.getViewerPanel().state().getCurrentTimepoint();
		//System.out.println(position);
		//this.rawRA.setPosition(new int[] {(int) position.getDoublePosition(0), (int)position.getDoublePosition(1), timepoint});
		//this.rawRA.get().set(this.rawRA.get().get()+(float)0.2);
		INDArray input = Nd4j.zeros(1, 1, 216, 216, 216);
		for(int x=0; x<216; x++) {
		    for(int y=0; y<216; y++) {
			for(int z=0; z<216; z++) {
			    rawRA.setPosition(new int[] {(int)(position.getFloatPosition(0)/2-108)+x, (int)(position.getFloatPosition(1)/2-108)+y, (int)(position.getFloatPosition(2)/2-108)+z});
			    input.putScalar(new int [] {1,1, x, y, z}, rawRA.get().get());
			}
		    }
		}
		Map<String,INDArray> inputs = new LinkedHashMap<>();
		inputs.put("input",input);
		Map<String, INDArray> exec = onnxRuntimeRunner.exec(inputs);
		INDArray output = exec.get("output");
		System.out.println(output.shapeInfoToString());
		
		RandomAccessibleInterval<FloatType> outputImg = ArrayImgs.floats(68,68,68);
		RandomAccess<FloatType> outputImgRA = outputImg.randomAccess();
		for(int x=0; x<68; x++) {
		    for(int y=0; y<68; y++) {
			for(int z=0; z<68; z++) {
			    outputImgRA.setPosition(new int[] {x, y, z});
			    outputImgRA.get().set((float) (output.getFloat(0,2,x,y,z)*128+127));
			}
		    }
		}
		
		AffineTransform3D t = new AffineTransform3D();
		System.out.println(position.getDoublePosition(0)+" "+position.getDoublePosition(1)+" "+position.getDoublePosition(2));
		t.translate(new double[] {position.getDoublePosition(0)-34,position.getDoublePosition(1)-34,position.getDoublePosition(2)-34});
		final N5Source source = new N5Source<>(
	        	new UnsignedByteType(),
	                "raw",
	                new RandomAccessibleInterval [] {outputImg},
	                new AffineTransform3D[] {t});

		//BdvFunctions.show(source);
		BdvOptions bdvOptions = BdvOptions.options().addTo(bdv);
		BdvFunctions.show(source, bdvOptions);
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
	RandomAccess<UnsignedByteType> rawRA = null;
	Bdv bdv = null;
	BdvHandle bdvHandle = null;
	OnnxRuntimeRunner onnxRuntimeRunner = null;
	public MouseMotionEventDemo(Bdv bdv, RandomAccess<UnsignedByteType> randomAccess, OnnxRuntimeRunner onnxRuntimeRunner) {
	    this.bdv = bdv;
	    this.bdvHandle = bdv.getBdvHandle();
	    this.rawRA = randomAccess;
	    this.onnxRuntimeRunner = onnxRuntimeRunner;
	}

	public void mouseDragged(MouseEvent e) {
	}

	public void mouseMoved(MouseEvent e) {
	    if (draw) {
		/*
		final RealPoint position = BdvUtils.getGlobalMouseCoordinates(bdvHandle);
		//final int timepoint = bdvHandle.getViewerPanel().state().getCurrentTimepoint();
		//System.out.println(position);
		//this.rawRA.setPosition(new int[] {(int) position.getDoublePosition(0), (int)position.getDoublePosition(1), timepoint});
		//this.rawRA.get().set(this.rawRA.get().get()+(float)0.2);
		INDArray input = Nd4j.zeros(1, 1, 216, 216, 216);
		for(int x=0; x<216; x++) {
		    for(int y=0; y<216; y++) {
			for(int z=0; z<216; z++) {
			    rawRA.setPosition(new int[] {(int)(position.getFloatPosition(0)-108)+x, (int)(position.getFloatPosition(1)-108)+y, (int)(position.getFloatPosition(1)-108)+z});
			    input.putScalar(new int [] {1,1,1, x, y, z}, rawRA.get().get()/255.0);
			}
		    }
		}
		Map<String,INDArray> inputs = new LinkedHashMap<>();
		inputs.put("input",input);
		Map<String, INDArray> exec = onnxRuntimeRunner.exec(inputs);
		INDArray output = exec.get("output");
		System.out.println(output.shapeInfoToString());
		
		RandomAccessibleInterval<FloatType> outputImg = ArrayImgs.floats(68,68,68);
		RandomAccess<FloatType> outputImgRA = outputImg.randomAccess();
		for(int x=0; x<68; x++) {
		    for(int y=0; y<68; y++) {
			for(int z=0; z<68; z++) {
			    outputImgRA.setPosition(new int[] {x, y, z});
			    outputImgRA.get().set(output.getFloat(0,0,x,y,z));
			}
		    }
		}
		
		AffineTransform3D t = new AffineTransform3D();
		t.scale(0.5);
		t.translate(new double[] {position.getDoublePosition(0)-34,position.getDoublePosition(1)-34,position.getDoublePosition(2)-34});
		final N5Source source = new N5Source<>(
	        	new UnsignedByteType(),
	                "raw",
	                new RandomAccessibleInterval [] {outputImg},
	                new AffineTransform3D[] {new AffineTransform3D()});

		//BdvFunctions.show(source);
		BdvOptions bdvOptions = BdvOptions.options().addTo(bdv);
		BdvFunctions.show(source, bdvOptions);
		*/
	    }
	}
    }

    public static OnnxRuntimeRunner loadOnnxModel(String filename) {
	OnnxRuntimeRunner onnxRuntimeRunner = OnnxRuntimeRunner.builder()
		.modelUri(new File(filename).getAbsolutePath())
		.build();
	return onnxRuntimeRunner;
    }

    public static void main(String[] args) throws Exception {

	/*int[] imgDim = new int[] {28,28};
	long[] bdvDim = new long[] {216, 216, 216};
	Img<FloatType> bdvImg = ArrayImgs.floats(bdvDim);
	//System.out.println(img.dataType());
	RandomAccess<FloatType> bdvImgRA = bdvImg.randomAccess();
	for(int x=0; x<bdvDim[0]; x++) {
	    for(int y=0; y<bdvDim[1]; y++) {
		for (int z=0; z<bdvDim[2]; z++) {
		    bdvImgRA.setPosition(new int[] {x, y, z} );
		    bdvImgRA.get().set((float)Math.random());
		}
	    }
	}*/
	String n5path = "/groups/cellmap/cellmap/data/jrc_mus-liver/jrc_mus-liver.n5";
        final N5Reader n5Reader = new N5FSReader(n5path);
	final RandomAccessibleInterval<UnsignedByteType> raw = N5Utils.openVolatile(n5Reader,"/volumes/raw/s1");

        final SharedQueue sharedQueue = new SharedQueue( 8 );
        AffineTransform3D t = new AffineTransform3D();
        t.scale(2);
        final N5Source source = new N5Source<>(
        	new UnsignedByteType(),
                "raw",
                new RandomAccessibleInterval [] {raw},
                new AffineTransform3D[] {t});

        final N5VolatileSource volatileSource = source.asVolatile(sharedQueue);
	//t.scale(2);
	//t.translate(new double[] {50, 50, 50});
	BdvSource bdv = BdvFunctions.show(volatileSource);//sourceTransform(t));//, Bdv.options().is2D().axisOrder(AxisOrder.XYT));
	
	//bdv.setDisplayRange(0,1);
	BdvHandle bdvHandle = bdv.getBdvHandle();
	//OnnxRuntimeRunner onnxRuntimeRunner = null;
	OnnxRuntimeRunner onnxRuntimeRunner = loadOnnxModel("/groups/scicompsoft/home/ackermand/Programming/deeplearning4j-examples/python/cellmap_model.onnx");
	//bdvHandle.getViewerPanel().getDisplay().addMouseMotionListener(new MouseMotionEventDemo(bdv, raw.randomAccess(), onnxRuntimeRunner));
	bdvHandle.getViewerPanel().getDisplay().addKeyListener(new KeyListenerDemo(bdv, raw.randomAccess(), onnxRuntimeRunner));

    }
}

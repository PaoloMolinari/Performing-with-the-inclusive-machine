(
s.options.numBuffers = 1024 * 16;
s.options.memSize = 8192 * 64;
s.options.numOutputBusChannels = 4;
//s.boot;
s.meter;
s.plotTree;

thisProcess.openUDPPort(57120);

s.waitForBoot{
	
	var synth, infosynth, paramtitle, spacetitle, mpparamtitle, graphtitle, acttitle;
	var ta, tb, tc, td, te, tf, meter;
	var win, xyslider, uv, mss, msmp, msdel, sdelta, mpdict, sldactivity;

	// Tha training will be done live or usin JSON
	var livetraining = false;

	// FluidDatasets for Mediapipe data (mpdata) (input of NN)
	// for Synth (output NN)
	var mpdata = FluidDataSet(s);
	var outdata = FluidDataSet(s);
	// Create array face data

	// number of values MediaPipe
	var numValues = 66;

	// Data from MediaPipe
	// adata comes from LIVE (live training)
	// bdata come from JSON (pictures training)
	var adata = Array.fill(numValues, 0);
	var adatasmooth = Array.fill(numValues, 0);
	var bdata = Array.fill(numValues, 0);

	// Variables to get the delta change
	// padata stores de previous data
	// deltas stores the data - pdata
	// sdeltas stores smooth deltas
	//activity stores the average of change od the sdeltas array

	var padata = Array.fill(numValues, 0);
	var deltas = Array.fill(numValues, 0);
	var sdeltas = Array.fill(numValues, 0);
	var activity = Array.fill(1, 0);


	// Buffer for inputs and outputs data
	// mpbuf, Buffer from MediaPipe
	// parambuf, Buffer to synth and control 
	var mpbuf = Buffer.alloc(s, numValues);
	var paramsbuf = Buffer.alloc(s, 78);

	var synthpbuf = Buffer.alloc(s, 66);
	var spaAPBus = Buffer.alloc(s, 2);
	var spaBPBus = Buffer.alloc(s, 2);
	var spaCPBus = Buffer.alloc(s, 2);
	var spaDPBus = Buffer.alloc(s, 2);
	var spaEPBus = Buffer.alloc(s, 2);
	var spaFPBus = Buffer.alloc(s, 2);

	var allBuf = Buffer.alloc(s, 78);
	
	//Buffer para Deltas
	var deltabuf = Buffer.alloc(s, numValues);

	var counter = 0;
	// Boolean and a integer variables
	// for predicting/not predicting
	var predicting = 0;
	var predict = false;

	// Boolean draw points or not
	var visibility = true;

	// Function to draw a circle
	var circle = {
		|x, y, r|
		Pen.fillColor = Color.new255(255, 0, 0, 255);
		Pen.strokeColor = Color.red;
		Pen.addArc(x@y + ((r@r)), r, 0, 2pi);
		Pen.fillStroke;
	};


	//Fluid Regressor Declaration 	
	var mlp = FluidMLPRegressor(s,
		[7],
		activation:FluidMLPRegressor.sigmoid,
		outputActivation:FluidMLPRegressor.sigmoid,
		maxIter: 1000,
		learnRate:0.1,
		batchSize:1,
		validation:0
	);

	s.sync;

	// OSC Message from MediaPipe Live

	NN.load(
		\moda,
		"/home/pemb/Music/Supercollider/nnRave/models/voice-multi-b2048-r48000-z11.ts",
		action: _.describe ); //11
	
	NN.load(
		\modb,
		"/home/pemb/Music/Supercollider/nnRave/models/guitar_iil_b2048_r48000_z16.ts",
		action: _.describe ); //16
	
	NN.load(
		\modc,
		"/home/pemb/Music/Supercollider/nnRave/models/mrp_strengjavera_b2048_r44100_z16.ts",
		action: _.describe); //8
	
	NN.load(
		\modd,
		"/home/pemb/Music/Supercollider/nnRave/models/birds_dawnchorus_b2048_r48000_z8.ts",
		action: _.describe); //8
	
	NN.load(
		\mode,
		"/home/pemb/Music/Supercollider/nnRave/models/percussion.ts",
		action: _.describe); // 4
	
	NN.load(
		\modf,
		"/home/pemb/Music/Supercollider/nnRave/models/crozzoli_bigensemblesmusic_18d.ts",
		action: _.describe); //18

	s.sync;

	OSCdef(\oscposes, {
		arg msg, mmsg;

		var smoothing = 0.9;	
		// clean the OSC path
		var incarray = msg[1..];
		adata.clear;

		// take out each thris element Z axis
		adata = incarray.reject({
			arg item, i;
			(i + 1) % 3 == 0 }
		);

		// Copy the Array to the Buffer
		//mpbuf.setn(0, adata);

		// Calculate the amount of movement and keep it in an array
		adata.do { |val, i|
			deltas[i] = val - padata[i];
			// Smooth delta
			sdeltas[i]  = (smoothing * sdeltas[i]) + ((1 - smoothing) * deltas[i]);

            adatasmooth[i] = (0.99 * padata[i]) + ((1 - 0.99) * adata[i]);
			
			
			// copy actual value to pdata
			padata[i] = val;
		};

		// Smooth mediPipe.
		
		// Copy the Array to the Buffer
		mpbuf.setn(0, adatasmooth);

		// Take the absolute and mean of the delta changes

		//activity is non 1 only when predicting

		activity[0] = sdeltas.abs.mean;
		//activity[0].postln;

		// Copy Array sdeltas to deltabuf
		deltabuf.setn(0, sdeltas);

		// If predict is true, asign the adata values
		// to the multiSlider msmp 
		if(predict == true, {
			//			{msmp.valueAction_(adata)}.defer;
			{msmp.valueAction_(adatasmooth)}.defer;
			//{msdel.valueAction_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0))}.defer;
			{sdelta.valueAction_(activity.linlin(0.0, 0.1, 0.0, 1.0))}.defer;
			defer{uv.refresh};
		});

		// If live training is True
		// asign the values of adata to MultiSlider msmp
		if(livetraining == true, {
			//			{msmp.valueAction_(adata)}.defer;
			{msmp.valueAction_(adatasmooth)}.defer;
			//{msdel.valueAction_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0))}.defer;
			{sdelta.valueAction_(activity.linlin(0.0, 0.1, 0.0, 1.0))}.defer;
			defer{ uv.refresh };
		});
	}, '/landmarks/0');

	//---GUI---//
	//---------//

	// Create Window
	Window.closeAll;
	win = Window("Training Window", Rect(10, 10, 1670, 350))
	.front
	.alwaysOnTop_(true);

	win.onClose_({
		s.freeAll;
		"pkill -f bodyPose.py".unixCmd;  // kill it on toggle off
	});

	// Text field to display info synth parameter
	paramtitle  = StaticText(win, Rect(10, 5, 200, 30))
	.string_("SYNTH ARGUMENTS")
	.align_(\left);

	// Text field to display info synth parameter
	mpparamtitle  = StaticText(win, Rect(590, 5, 200, 30))
	.string_("MEDIAPIPE DATA")
	.align_(\left);

	// Text field to display info synth parameter
	graphtitle  = StaticText(win, Rect(1000, 5, 200, 30))
	.string_("MEDIAPIPE BODY POINT")
	.align_(\left);

	// Text field to display info synth parameter
	infosynth  = StaticText(win, Rect(200, 5, 200, 30))
	.string_(" ")
	.align_(\left);

	// Text field to display info synth parameter
	spacetitle  = StaticText(win, Rect(420, 5, 200, 30))
	.string_("SPATIALIZATION")
	.align_(\left);

	// Multislider Synth arguments 
	mss = MultiSliderView(win, Rect(10, 30, 400, 300))
	.elasticMode_(1)
	.isFilled_(1)
	.showIndex_(true)
	.action_({
		arg ms;
		paramsbuf.setn(0, ms.value);
			infosynth.string_("Slider: %  Value: %".format(
			ms.index,
			ms.value[ms.index].round(0.01)
			));	
	}).value_(0.0.dup(65)); //65

	//Multislider MediaPipe from OSC/JSON

	msmp = MultiSliderView(win, Rect(590, 30, 400, 300));

	// If Live Training is true tha value of msmp is adata 
	//	if(livetraining == true){msmp.value_(adata)};
	if(livetraining == true){msmp.value_(adatasmooth)};

	// If Live Training is false and predict is false tha value of msmp is bdata 
	if((livetraining == false) && (predict == false), {msmp.value_(bdata)});

	msmp.elasticMode_(1);
	msmp.isFilled_(0);
	msmp.drawLines_(0);
	msmp.drawRects_(1);
	msmp.showIndex = false;
	msmp.doAction;

	// Definition of the UserView
	// to draw the points of MediaPipe
	uv = UserView(win, Rect(1000, 30, 400, 300));
	uv.background_(Color.black);
	uv.drawFunc = { |uvview|
		// If visibility is True
		if(visibility == true){
			if((livetraining == true)) {
				/*
				adata.pairsDo{ |a, b|
					circle.value(
						a * uv.bounds.width,
						b * uv.bounds.height,
						2
					);
				};
				*/
				adatasmooth.pairsDo{ |a, b|
					circle.value(
						a * uv.bounds.width,
						b * uv.bounds.height,
						2
					);
				};
			};
			
			if((predict == true)) {
				/*
				adata.pairsDo{ |a, b|
					circle.value(
						a * uv.bounds.width,
						b * uv.bounds.height,
						2
					);
					};
				*/

				adatasmooth.pairsDo{ |a, b|
					circle.value(
						a * uv.bounds.width,
						b * uv.bounds.height,
						2
					);
				};
			};
	
			if((livetraining == false) && (predict == false)) {
				bdata.pairsDo{ |a, b|
					circle.value(
						a * uv.bounds.width,
						b * uv.bounds.height,
						2
					);
				};
			};
		};
	};

	//Multislider DELTA OSC/JSON
	/*
		msdel = MultiSliderView(win, Rect(810, 10, 400, 300));
		msdel.value_(sdeltas.linlin(-0.1, 0.1, 0.0, 1.0));
		msdel.elasticMode_(1);
		msdel.isFilled_(0);
		msdel.doAction; 
	*/

	sdelta = MultiSliderView(win, Rect(1410, 30, 20, 300));
	//sdelta.size(1);
	sdelta.value_(activity.linlin(0.0, 0.001, 0.0, 1.0));
	sdelta.elasticMode_(1);
	sdelta.isFilled_(1);
	sdelta.doAction;

	//sldactivity = Slider(win, Rect(1210, 10, 10, 300)).value_(activity);

	// load data file
	Button(win,Rect(1440, 30, 100, 20))
	.states_([["Load Data File"]])
	.action_{
		FileDialog({
			arg path, f;
			path.postln;
			f = File( path, "r");
			mpdict = f.readAllString.parseJSON;
			bdata = mpdict.at("%".format(counter)).at("landmarks").asFloat;
			bdata = bdata.reject({ arg item, i; (i + 1) % 3 == 0 });
			{msmp.valueAction_(bdata)}.defer;
			mpbuf.setn(0, bdata);
			defer{ uv.refresh };
		}, {} , 1, 0, true);
	};

	//ADD DATA TO THE DATASETS

	Button(win,Rect(1440,60,100,20))
	.states_([["Add Points"]])
	.action_{
		var id = "point-%".format(counter);
	
		if(livetraining == false) {
			if(predict == false){
				bdata = mpdict.at("%".format(counter)).at("landmarks").asFloat;
				bdata = bdata.reject({ arg item, i; (i + 1) % 3 == 0 });
				{msmp.valueAction_(bdata)}.defer;
				mpbuf.setn(0, bdata);
				defer{ uv.refresh };
			};
		};
		
		mpdata.addPoint(id, mpbuf);
		outdata.addPoint(id, paramsbuf);
		counter = counter + 1;
		mpdata.print;
		outdata.print;		
	};

	//SAVE DATA POINTS COLLECTED TO A FILE

	Button(win,Rect(1440,90,100,20))
	.states_([["Save Data"]])
	.action_{
		FileDialog({
			arg folder;
			folder.postln;
			mpdata.write(folder+/+"xydata.json");
			outdata.write(folder+/+"paramsdata.json");
		}, {}, 2, 0, true);
	};

	//RESTORE DATA POINTS COLLECTED FROM A FILE

	Button(win,Rect(1440,120,100,20))
	.states_([["Load Data"]])
	.action_{
		FileDialog({
			arg folder;
			mpdata.read(folder+/+"xydata.json");
			outdata.read(folder+/+"paramsdata.json");
		}, fileMode: 2, acceptMode: 0, stripResult: true);
	};

	//TRAIN MODEL

	Button(win,Rect(1440,150,100,20))
	.states_([["Train"]])
	.action_{
		mlp.fit(mpdata, outdata,{
			arg loss;
			loss.postln;
		});
	};

	//SAVE TRAINED MODEL

	Button(win,Rect(1440, 180, 100, 20))
	.states_([["Save MLP"]])
	.action_{
		Dialog.savePanel({
			arg path;
			if(PathName(path).extension != "json"){
				path = "%.json".format(path);
			};
			mlp.write(path);
		});
	};

	//LOAD SAVED MODEL
	Button(win,Rect(1440, 210, 100, 20))
	.states_([["Load MLP"]])
	.action_{
		Dialog.openPanel({
			arg path;
			mlp.read(path);
		});
	};

	//INFERENCE OR TRAINING SWITCH
	Button(win,Rect(1440, 240, 100, 20))
	.states_([["Not Predicting"],["Predicting"]])
	.action_{
		arg but;
		if(but.value == 1){predict = true}{predict == false};
		
		synth.set(\predicting, but.value);
	};

	Button(win,Rect(1440, 270, 100, 20))
	.states_([["File Training"],["Live Training"]])
	.action_{
		arg msg;
		livetraining = msg.value.asBoolean;
		livetraining.postln;
	};

	Button(win,Rect(1440, 300, 100, 20))
	.states_([["Show Data"],["Hide Data"]])
	.action_{
		arg msg;
		visibility = msg.value.asBoolean;
		visibility.postln;
	};

	Button(win,Rect(1560, 30, 100, 100))
	.states_([
        ["Start VTrack", Color.white, Color.green(0.6)],
        ["Stop VTrack",  Color.white, Color.red(0.6)],
    ])
	.action_{
		arg msg;
		if (msg.value == 1) {
			~pid = "bash -c 'conda run -n mediapipe python /home/pemb/Documents/Proyectos/Mime_interactive/Musical_Instrument/mpPoseLive.py'".unixCmd;	
        } {
            "pkill -f bodyPose.py".unixCmd;  // kill it on toggle off
        };
	};

	meter = ServerMeterView(s, win, Point(1540, 130), 0, 4);

	ta = Slider2D(win, Rect(420, 30, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(66, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(67, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	tb = Slider2D(win, Rect(500, 30, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(68, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(69, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	tc = Slider2D(win, Rect(420, 140, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(70, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(71, sl.y.linlin(0, 1, -1, 1));
	});

	td = Slider2D(win, Rect(500, 140, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(72, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(73, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	te = Slider2D(win, Rect(420, 250, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(74, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(75, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	tf = Slider2D(win, Rect(500, 250, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(76, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(77, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	~synthbus = Bus.audio(s, 2);

	synth =  {
		arg predicting = 0;
		var sig, outsig, mp, trig, val;
		var lata, latb, latc, latd, late, latf;
		/*
			var apx = 0.5;
			var apy = 0.5;
			var bpx = 0.5;
			var bpy = 0.5;
			var cpx = 0.5;
			var cpy = 0.5;
			var dpx = 0.5;
			var dpy = 0.5;
			var epx = 0.5;
			var epy = 0.5;
			var fpx = 0.5;
			var fpy = 0.5;
		*/

		var apx, apy, bpx, bpy, cpx, cpy, dpx, dpy, epx, epy, fpx, fpy;
		var laa, lab, lac, lad, lae, laf;
		var ga, gb, gc, gd, ge, gf;
		var gaa, gbb, gcc, gdd, gee, gff;
		var aenv, benv, cenv, denv, eenv, fenv;
		var as, bs, cs, ds, es, fs;
		var asm, bsm, csm, dsm, esm, fsm;


		var act = activity[0];
		
		var mod = 2;
		var mrate = 0.1;

		var amp = 1.0;
		
		//comvert the mpbuf to control 
		mp = FluidBufToKr.kr(mpbuf);
		trig = Mix(Changed.kr(mp));

		//only when is triggered.
		mlp.kr(trig * predicting, mpbuf, paramsbuf);

		val = FluidBufToKr.kr(paramsbuf);

		//paramsbuf.poll;

		//Define a OSC message to be send with the
		//values of the paramsbuffer to the address
		// /paramsChnaged, everytome the values in
		//paramsbuf change. This is values from the
		//sound server to the language interpreter
		SendReply.kr(Mix(Changed.kr(val[0..65])), "/paramsChanged", val[0..65]);
		SendReply.kr(Mix(Changed.kr(val[66..67])), "/paramsSpaAChanged", val[66..67]);
		SendReply.kr(Mix(Changed.kr(val[68..69])), "/paramsSpaBChanged", val[68..69]);
		SendReply.kr(Mix(Changed.kr(val[70..71])), "/paramsSpaCChanged", val[70..71]);
		SendReply.kr(Mix(Changed.kr(val[72..73])), "/paramsSpaDChanged", val[72..73]);
		SendReply.kr(Mix(Changed.kr(val[74..75])), "/paramsSpaEChanged", val[74..75]);
		SendReply.kr(Mix(Changed.kr(val[76..77])), "/paramsSpaFChanged", val[76..77]);
	
		/*
			lata = FluidBufToKr.kr(paramsbuf, 0, 11); // 11: 0-10
			latb = FluidBufToKr.kr(paramsbuf, 11, 16); // 16: 11- 26
			latc = FluidBufToKr.kr(paramsbuf, 27, 8); // 8: 27 - 34  
			latd = FluidBufToKr.kr(paramsbuf, 35, 8); // 8: 35 - 42
			late = FluidBufToKr.kr(paramsbuf, 43, 4); // 4: 43 - 46
			latf = FluidBufToKr.kr(paramsbuf, 47, 18); // 18: 47 - 64
		*/
		
		lata = val[0..10]; //FluidBufToKr.kr(parambuf, 0, 11); // 11: 0-10
		latb = val[11..26]; //FluidBufToKr.kr(parambuf, 11, 16); // 16: 11- 26
		latc = val[27..34]; //FluidBufToKr.kr(parambuf, 27, 8); // 8: 27 - 34  
		latd = val[35..42]; //FluidBufToKr.kr(parambuf, 35, 8); // 8: 35 - 42
		late = val[43..46]; //FluidBufToKr.kr(parambuf, 43, 4); // 4: 43 - 46
		latf = val[47..64]; //FluidBufToKr.kr(parambuf, 47, 18); // 18: 47 - 64

		
		/*
			apx = FluidBufToKr.kr(paramsbuf, 65, 1);
			apy = FluidBufToKr.kr(paramsbuf, 66, 1);
			bpx = FluidBufToKr.kr(paramsbuf, 67, 1);
			bpy = FluidBufToKr.kr(paramsbuf, 68, 1);
			cpx = FluidBufToKr.kr(paramsbuf, 69, 1);
			cpy = FluidBufToKr.kr(paramsbuf, 70, 1);
			dpx = FluidBufToKr.kr(paramsbuf, 71, 1);
			dpy = FluidBufToKr.kr(paramsbuf, 72, 1);
			epx = FluidBufToKr.kr(paramsbuf, 73, 1);
			epy = FluidBufToKr.kr(paramsbuf, 74, 1);
			fpx = FluidBufToKr.kr(paramsbuf, 75, 1);
			fpy = FluidBufToKr.kr(paramsbuf, 76, 1);
		*/
	
		apx = val[66]; //FluidBufToKr.kr(parambuf, 65, 1);
		apy = val[67]; //FluidBufToKr.kr(parambuf, 66, 1);

		bpx = val[68]; //FluidBufToKr.kr(parambuf, 67, 1);
		bpy = val[69]; //FluidBufToKr.kr(parambuf, 68, 1);

		cpx = val[70]; //FluidBufToKr.kr(parambuf, 69, 1);
		cpy = val[71]; //FluidBufToKr.kr(parambuf, 70, 1);

		dpx = val[72]; //FluidBufToKr.kr(parambuf, 71, 1);
		dpy = val[73]; //FluidBufToKr.kr(parambuf, 72, 1);

		epx = val[74]; //FluidBufToKr.kr(parambuf, 73, 1);
		epy = val[75]; //FluidBufToKr.kr(parambuf, 74, 1);

		fpx = val[76]; //FluidBufToKr.kr(parambuf, 75, 1);
		fpy = val[77]; //FluidBufToKr.kr(parambuf, 76, 1);
	
		/*
		laa = K2A.ar(Lag.kr(lata, 0.2));
		lab = K2A.ar(Lag.kr(latb, 0.2));
		lac = K2A.ar(Lag.kr(latc, 0.2));
		lad = K2A.ar(Lag.kr(latd, 0.2));
		lae = K2A.ar(Lag.kr(late, 0.2));
		laf = K2A.ar(Lag.kr(latf, 0.2));
		*/

		laa = K2A.ar(lata);
		lab = K2A.ar(latb);
		lac = K2A.ar(latc);
		lad = K2A.ar(latd);
		lae = K2A.ar(late);
		laf = K2A.ar(latf);

		laa.poll;
		
		ga = Changed.kr(Mix.kr(lata), 0.01);
		gb = Changed.kr(Mix.kr(latb), 0.01);
		gc = Changed.kr(Mix.kr(latc), 0.01);
		gd = Changed.kr(Mix.kr(latd), 0.01);
		ge = Changed.kr(Mix.kr(late), 0.01);
		gf = Changed.kr(Mix.kr(latf), 0.01);
	
		gaa = Changed.kr(ga, 0.01);
		gbb = Changed.kr(gb, 0.01);
		gcc = Changed.kr(gc, 0.01);
		gdd = Changed.kr(gd, 0.01);
		gee = Changed.kr(ge, 0.01);
		gff = Changed.kr(gf, 0.01);


		//non continuos only when predicting
		
		aenv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gaa, doneAction: 0)]
		);
		
		benv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gbb, doneAction: 0)]
		);
		
		cenv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gcc, doneAction: 0)]
		);
		
		denv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gdd, doneAction: 0)]
		);
		
		eenv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gee, doneAction: 0)]
		);
		
		fenv = Select.kr(
			predicting,
			[1, EnvGen.kr(Env.asr(0.5, 1.0, 1.0), gff, doneAction: 0)]
		);

		/*
			var mla = lata.collect{|l|
			LFNoise2.ar(mrate).range(-1 * mod, mod) + l}; 
			var mlb = latb.collect{|l|
			LFNoise2.ar(mrate).range(-1 * mod, mod) + l}; 
		*/

		//laa.poll;

		
		as = NN(\moda, \decode).ar(laa, 8192);
		bs = NN(\modb, \decode).ar(lab, 8192);
		cs = NN(\modc, \decode).ar(lac, 8192);
		ds = NN(\modd, \decode).ar(lad, 8192);
		es = NN(\mode, \decode).ar(lae, 8192);
		fs = NN(\modf, \decode).ar(laf, 8192);

		as = as * aenv;
		bs = bs * benv;
		cs = cs * cenv;
		ds = ds * denv;
		es = es * eenv;
		fs = fs * fenv;

		asm = Pan4.ar(Mix.ar(as), apx, apy, 1);
		bsm = Pan4.ar(Mix.ar(bs), bpx, bpy, 1);
		csm = Pan4.ar(Mix.ar(cs), cpx, cpy, 1);
		dsm = Pan4.ar(Mix.ar(ds), dpx, dpy, 1);
		esm = Pan4.ar(Mix.ar(es), epx, epy, 1);
		fsm = Pan4.ar(Mix.ar(fs), fpx, fpy, 1);

		sig = asm + bsm + csm + dsm + esm + fsm;

		outsig  = sig.collect { |ch|
			FreeVerb.ar(
				ch,
				LFNoise2.kr(1).range(0, 1),
				LFNoise2.kr(1).range(0, 1),
				LFNoise2.kr(1).range(0, 1)
			) * 2;
		};

		//act.poll;

		//amp = Select.kr(predicting, [1, activity[0]]);

		//amp.poll;
		
		//outsig = outsig * amp;
		outsig
}.play;




/*
//SYNTH
synth = {
	arg predicting = 0;
	var in, latent, modLatent, prior, rlmod = 20;
	var osc1, osc2, feed1, feed2, base1=69, base2=69, base3 = 130, loud1, loud2, sig;
	var val, mp, trig;

	//comvert the mpbuf to control 
	mp = FluidBufToKr.kr(mpbuf);
	//trigger if the control changes
	trig = Mix(Changed.kr(mp));
	//The trigger is activated when the argument
	//predicting is 1. Otherwise the trigger will
	//be always 0, therefore not triggering.
	//The mlp claculates the outpus of the nn
	//only when is triggered.
	mlp.kr(trig * predicting, mpbuf, paramsbuf);
	val = FluidBufToKr.kr(paramsbuf);
	//Define a OSC message to be send with the
	//values of the paramsbuffer to the address
	// /paramsChnaged, everytome the values in
	//paramsbuf change. This is values from the
	//sound server to the language interpreter
	SendReply.kr(Mix(Changed.kr(val)), "/paramsChanged", val);
	
	
	#feed2, feed1 = LocalIn.ar(2);

	//only for testing is we get the data from MP. 
	//	FluidBufToKr.kr(~mpbuf).poll;

	loud1 = FluidLoudness.kr(feed1,[\loudness],1,0,hopSize:64);
	loud2 = FluidLoudness.kr(feed2,[\loudness],1,0,hopSize:64);

	osc1 = SinOsc.ar((((feed1 * val[0]) +  val[1]) * base1).midicps,mul:(val[2] * 50).dbamp).atan;
	osc1 = MoogFF.ar(osc1,(base3-(val[3]*(loud2.clip(-120,0)+120))).lag(128/44100).midicps,val[4]*3.5);

	osc2 = SinOsc.ar((((feed2 * val[5]) +  val[6]) * base2).midicps,mul: (val[7] * 50).dbamp).atan;
	osc2 = MoogFF.ar(osc2,(base3-(val[8]*(loud1.clip(-120,0)+120))).lag(128/44100).midicps,val[9]*3.5);

	//		Out.ar(~synthbus,LeakDC.ar([osc1,osc2],mul:0.1));

	LocalOut.ar([osc1,osc2]);
	
sig = [osc1,osc2] * 0.1;

	/*
    // encode sound input to RAVE latent space
    latent = NN(\ravePerc, \encode).ar(sig);
    // add a random modulation to every dimension
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(val[10] * rlmod).range(-0.5, 0.5)
    };
    // resynthesize modulated latents
	//    NN(\ravePerc, \decode).ar(modLatent);

	
	FreeVerb.ar(NN(\ravePerc, \decode).ar(modLatent), LFNoise2.kr(1).range(0,1), LFNoise2.kr(1).range(0,1), LFNoise2.kr(1).range(0,1));
	*/
}.play;
	*/

//OSCdef to receive the masse from the audio server
//to the language interpreter
OSCdef(\paramsChanged,{
	arg msg;
	defer{
		msg.postln;
		//mss.value_(msg[3..]);  // up to 65 nd then the rest to control each xy slider
	};
},"/paramsChanged");

	//};

OSCdef(\paramsSpaAChanged,{
	arg msg;
	defer{
		//msg.postln;
		ta.x_(msg[3]);
		ta.y_(msg[4]);
	};
},"/paramsSpaAChanged");

OSCdef(\paramsSpaBChanged,{
	arg msg;
	defer{
		//msg.postln;
		tb.x_(msg[3]);
		tb.y_(msg[4]);
	};
},"/paramsSpaBChanged");

	OSCdef(\paramsSpaCChanged,{
	arg msg;
	defer{
		//msg.postln;
		tc.x_(msg[3]);
		tc.y_(msg[4]);
	};
},"/paramsSpaCChanged");

	OSCdef(\paramsSpaDChanged,{
	arg msg;
	defer{
		//msg.postln;
		td.x_(msg[3]);
		td.y_(msg[4]);
	};
},"/paramsSpaDChanged");

	OSCdef(\paramsSpaEChanged,{
	arg msg;
	defer{
		//msg.postln;
		te.x_(msg[3]);
		te.y_(msg[4]);
	};
},"/paramsSpaEChanged");

	OSCdef(\paramsSpaFChanged,{
	arg msg;
	defer{
		//msg.postln;
		tf.x_(msg[3]);
		tf.y_(msg[4]);
	};
},"/paramsSpaFChanged");



};

)

s.meter;
s.plotTree;




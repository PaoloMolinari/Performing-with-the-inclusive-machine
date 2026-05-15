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

	// to select multislider with the keyboard
	var selectedIndex = 0; // currently selected slider
	var label;
	
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
		"pkill -f mpPoseLive.py".unixCmd;  // kill it on toggle off
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
		paramsbuf.setn(6, ms.value);
			infosynth.string_("Slider: %  Value: %".format(
			ms.index,
			ms.value[ms.index].round(0.01)
			));	
	}).value_(0.0.dup(21)); //65


	
	//label = StaticText(win, Rect(10, 390, 380, 20));
	//label.string = "Slider: % | Value: %".format(
	//	selectedIndex, mss.value[selectedIndex].round(0.01));



	//change multislider with arrow keys
	
	win.view.keyDownAction = { |view, char, mod, unicode, keycode|
		switch(keycode,
			65361, {selectedIndex = (selectedIndex - 1).max(0) },
			65363, {selectedIndex = (selectedIndex + 1).min(mss.size - 1) },
			65362, {
				var vals =  mss.value.copy;
				vals[selectedIndex] = (vals[selectedIndex] + 0.01).min(1.0);
				mss.value_(vals);
				paramsbuf.setn(6, vals);
				mss.refresh;
				infosynth.string_("Slider: %  Value: %".format(
					selectedIndex,
					vals[selectedIndex].round(0.01)
			));	
			},
			65364, {
            var vals =  mss.value.copy;
				vals[selectedIndex] = (vals[selectedIndex] - 0.01).max(0.0);
				mss.value_(vals);
				paramsbuf.setn(6, vals);
				mss.refresh;
				infosynth.string_("Slider: %  Value: %".format(
					selectedIndex,
					vals[selectedIndex].round(0.01)
			));	

			},
		);

		mss.index = selectedIndex;
		label.string = "Slider: % | Value: %".format(
			selectedIndex, mss.value[selectedIndex].round(0.01));
		mss.refresh;
};

	
	
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
            "pkill -f mpPoseLive.py".unixCmd;  // kill it on toggle off
        };
	};

	meter = ServerMeterView(s, win, Point(1540, 130), 0, 4);

	ta = Slider2D(win, Rect(420, 30, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(0, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(1, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	tb = Slider2D(win, Rect(500, 30, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(2, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(3, sl.y.linlin(0, 1, -1, 1));
		//	[\sliderX, sl.x, \sliderY, sl.y].postln;
	});

	tc = Slider2D(win, Rect(420, 140, 80, 80))
	.x_(0.5) // initial location of x
	.y_(0.5)   // initial location of y
	.thumbSize_(10)
	.action_({ |sl|
		paramsbuf.set(4, sl.x.linlin(0, 1, -1, 1));
		paramsbuf.set(5, sl.y.linlin(0, 1, -1, 1));
	});

	/*
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
	*/
	//	~synthbus = Bus.audio(s, 2);

	synth =  {
		arg predicting = 0;

		//main output
		var main_sig = 0;
		
		//vars envelopes
		var atk = 0.01;
		var sus = 1.0;
		var rel = 3;

		// Instrs. Specific variables //

		// Instr. A (simple AM) //
		var a_carrier, a_modulator, a_env, a_data;
		var a_gate = 0;
		var a_freq = 440;
		var a_modfreq = 2;
		var a_amp = 0.2;
		var a_px = 0.5;
		var a_py = 0.5;
		var a_osig = 0;

		// Instr. B (complex AM) //

		var b_carrier, b_modulator, b_env, b_data;
		var b_gate = 0;
		var b_freq = 440;
		var b_modfreq = 2;
		var b_amp = 0.2;
		var b_px = 0.5;
		var b_py = 0.5;
		var b_osig = 0;

		// Instr. C (drone FM)

		var c_sig, c_car1, c_mod1, c_car2, c_mod2, c_car3, c_mod3, c_env, c_index, c_data;
		var c_f1Freq = 0.1;
		var c_f2Freq = 1.1;
		var c_f3Freq = 2.2;
		var degree = 1;
		var c_nFreq = 440;
		var c_m1Ratio = 1;
		var c_m2Ratio = 2;
		var c_m3Ratio = 3;
		var c_amp = 0.1;
		var c_ampM = 0.84;
		var c_gate = 0;
		var c_px = 0.5;
		var c_py = 0.5;
		var c_osig = 0;

		var val, mp, trig;
		
		// neural network

		mp = FluidBufToKr.kr(mpbuf);
		trig = Mix(Changed.kr(mp));
		mlp.kr(trig * predicting, mpbuf, paramsbuf);
		val = FluidBufToKr.kr(paramsbuf);

		
		SendReply.kr(
			predicting * Mix(Changed.kr(val[0..14])),
			"/paramsChanged",
			val[6..19]);
		
		SendReply.kr(
			predicting * Mix(Changed.kr(val[15..16])),
			"/paramsSpaAChanged",
			val[0..1]);
		
		SendReply.kr(
			predicting * Mix(Changed.kr(val[17..18])),
			"/paramsSpaBChanged",
			val[2..3]);

		SendReply.kr(
			predicting * Mix(Changed.kr(val[19..20])),
			"/paramsSpaCChanged",
			val[4..5]);

		a_data = val[6..8];
		a_gate = Changed.kr(Mix.kr(a_data), 0.01);
		a_freq = a_data[0].linexp(0.01, 1.0, 20.0, 1000.0);
		a_modfreq = a_data[1].linexp(0.01, 1.0, 20.0, 1000.0);
		a_amp = a_data[2];

		//a_data[2].poll;
		
		
		b_data = val[9..11];
		b_gate = Changed.kr(Mix.kr(b_data), 0.01);
		b_freq = b_data[0].linexp(0.01, 1.0, 20.0, 1000.0);
		b_modfreq = b_data[1].linexp(0.01, 1.0, 20.0, 1000.0);
		b_amp = b_data[2];

		c_data = val[12..20];
		c_gate = Changed.kr(Mix.kr(c_data), 0.01);
		c_f1Freq = c_data[0].linlin(0, 1, 0.0, 5).round(0.1);
		c_f2Freq = c_data[1].linlin(0, 1, 0.0, 5).round(0.1);
		c_f3Freq = c_data[2].linlin(0, 1, 0.0, 5).round(0.1);
		//degree = c_data[3].linlin(0, 1, 0, 7).round(1);
		//c_nFreq = degree.degreeToKey(Scale.major.degrees, 12).midicps;
		c_nFreq = c_data[3].linlin(0, 1, 0.0, 1000).round(0.1);
		c_m1Ratio = c_data[4].linlin(0, 1, 0.0, 5).round(0.1);
		c_m2Ratio = c_data[5].linlin(0, 1, 0.0, 5).round(0.1);
		c_m3Ratio = c_data[6].linlin(0, 1, 0.0, 5).round(0.1);
		c_amp = c_data[7].linlin(0, 1, 0.0, 1);
		c_ampM = c_data[8].linlin(0, 1, 0.0, 0.84);

		a_px = val[0];
		a_py = val[1];
		b_px = val[2];
		b_py = val[3];
		c_px = val[4];
		c_py = val[5];

		//c_px.poll;
		
		// Inst. A //
		a_env = Select.kr(predicting, [1, EnvGen.kr(
			Env.asr(
				attackTime: atk,
				releaseTime: rel),
			gate: a_gate,
			doneAction: 0)]);

		a_modulator = SinOsc.ar(a_modfreq).range(0, 1);
		a_carrier = SinOsc.ar(freq: a_freq, mul: a_env * a_modulator);
		a_carrier = a_carrier * a_amp;
		a_osig = Pan4.ar(a_carrier, a_px, a_py, 1);
		
		//Instr. B
		
		b_env = Select.kr(predicting, [1, EnvGen.kr(
			Env.asr(attackTime: atk, releaseTime: rel),
			gate: b_gate,
			doneAction: 0)]);
	
		b_modulator = SinOsc.ar(b_modfreq).range(0, 1);
		b_freq = b_freq * LFNoise2.kr([2, 1/2, 1/3, 1/4, 1]).range(0.98, 1.01);
		b_carrier = Pulse.ar(freq: b_freq, mul: b_env * b_modulator);
		b_carrier = Mix.ar(b_carrier);
		b_carrier = LPF.ar(b_carrier, Line.kr(10000, 1000, atk));
		b_carrier = b_carrier * b_amp;
		b_osig = Pan4.ar(b_carrier, b_px, b_py, 1);

		//Instr.C

		c_index = LFNoise1.kr(0.1).range(2, 20);
		//index = amp.linlin(0.01, 0.1, 2.0, 6.0);
		
		c_env = Select.kr(predicting, [1, EnvGen.kr(
			Env.asr(attackTime: atk, releaseTime: rel),
			gate: c_gate,
			doneAction: 0)]);
			
		c_index = c_index * c_env;
		
		c_mod1 = SinOsc.ar(
			[c_nFreq * c_m1Ratio, c_nFreq + 0.7 * c_m1Ratio],
			mul: c_nFreq * c_m1Ratio * c_index);
		c_mod1 = c_mod1 * c_ampM;
		c_car1 = SinOsc.ar(c_f1Freq + c_mod1);

		c_mod2 = SinOsc.ar(
			[c_nFreq * c_m2Ratio, c_nFreq + 1.4 * c_m2Ratio],
			mul: c_nFreq * c_m2Ratio * c_index);
		c_mod2 = c_mod2 * c_ampM;
		c_car2 = SinOsc.ar(c_f2Freq + c_mod2);

		c_mod3 = SinOsc.ar(
			[c_nFreq * c_m3Ratio, c_nFreq + 0.35 * c_m3Ratio],
			mul: c_nFreq * c_m3Ratio * c_index);
		c_mod3 = c_mod3 * c_ampM;
		c_car3 = SinOsc.ar(c_f3Freq + c_mod3) * 0.4;
		
		c_sig = Mix.ar((c_car1 + c_car2 + c_car3)) * c_env * c_amp;
		
		c_osig = Pan4.ar(c_sig, c_px, c_py, 1);

		b_osig.poll;

		main_sig = a_osig + b_osig + c_osig;
		main_sig
	}.play;




	
//OSCdef to receive the masse from the audio server
//to the language interpreter
OSCdef(\paramsChanged,{
	arg msg;
	defer{
		//msg.postln;
		mss.value_(msg[3..]);  // up to 65 nd then the rest to control each xy slider
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

	/*
	
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

	*/

};

)


s.plotTree;

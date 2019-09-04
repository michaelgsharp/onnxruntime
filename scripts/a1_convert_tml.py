import tml_pb2
import onnx
from onnx import numpy_helper
import numpy as np 
from PIL import Image

im = Image.open('n01440764_2708.png')
r,g,b = im.split()
im = Image.merge("RGB", (b, g, r))
im_np = np.array(im)
im_np = np.transpose(im_np, (2, 0, 1))
print(im_np.shape)
#im_np = im_np / np.float32(255.0)
im_np = im_np / np.float32(1.0)
im_np = np.reshape(im_np,(1,3,227,227))
with  open("input_0.pb", "wb") as f:
  t = numpy_helper.from_array(im_np)
  t.name = 'image'
  f.write(t.SerializeToString())

tensor = onnx.TensorProto()
tensor.dims.extend([1,1])
tensor.data_type = onnx.TensorProto.STRING
tensor.string_data.extend([b"electric ray, crampfish, numbfish, torpedo"])
with  open("output_0.pb", "wb") as f:
   f.write(tensor.SerializeToString())
exit(0)



data = tml_pb2.TraditionalMLData()
body = tml_pb2.VectorMapStringToFloat()
a = data.vector_map_string_to_float.v.add()
predict = [2.71168598687e-09,5.67120821415e-08,1.30748333049e-07,6.67020643164e-08,1.64903369182e-07,1.24376438038e-08,6.07931838204e-08,5.94040329815e-07,1.9677232288e-08,7.78654566602e-07,3.93363031037e-09,8.84788242672e-09,1.40374312174e-09,8.06229195405e-07,4.53386519439e-08,3.85078564591e-08,5.35102984145e-08,2.91113488871e-09,7.76130892888e-09,6.22909945847e-09,4.65789575799e-08,2.47237004203e-08,4.25756185862e-09,1.58231883152e-09,1.12905219396e-07,4.66058871496e-08,1.76674546992e-07,3.34203633656e-08,1.90582998272e-08,7.21836687645e-11,1.43815892528e-09,1.65886433479e-06,7.26584019617e-08,3.91706889147e-09,7.95341819071e-08,1.70883769357e-08,1.05568247477e-07,3.55603049229e-07,4.83559929876e-08,5.46459721917e-09,1.42390099711e-08,1.16959313345e-07,3.49147839529e-08,2.2379539999e-08,4.01320354726e-09,1.89285334073e-08,3.49557645052e-08,2.40290756182e-08,1.70673999378e-08,1.26642207832e-08,1.47623504532e-08,5.84540913451e-08,4.39681251407e-08,4.76101405411e-07,4.8063217406e-08,4.3814509354e-08,4.9495923804e-08,5.49574563635e-09,8.96350271784e-10,1.07739744237e-07,1.41475302371e-05,6.17217743581e-09,1.36976501253e-08,1.29911052227e-06,1.78098495951e-08,3.60182522696e-10,6.85327405936e-10,4.49526318391e-08,4.8020221584e-05,1.37468617822e-07,1.10776197104e-08,1.95191589114e-07,1.57540540613e-08,2.85483370277e-09,6.15228188394e-08,2.93300068677e-08,4.30104307725e-08,5.7111302354e-09,4.48012116294e-08,3.62643142182e-08,8.19315282286e-09,9.36878947755e-08,5.49143166495e-08,2.61032595539e-09,2.17875655295e-08,1.23159749066e-08,1.69687142115e-09,1.14779274973e-09,1.43426710508e-08,3.92813063854e-07,2.39581311234e-07,5.63049127322e-07,3.53800544417e-10,8.0364914723e-09,7.15204384605e-09,2.01255065946e-08,1.20118963665e-08,1.41229481443e-08,4.76718842179e-09,2.53234633263e-09,4.00334121409e-09,3.04685414676e-07,3.42311295753e-07,1.0141967266e-07,3.56409906033e-08,8.9079840393e-08,2.76367984497e-07,1.56356378511e-07,2.51592595646e-08,4.34892744039e-09,4.54952520101e-09,5.21439469381e-09,4.0580712124e-09,2.67666089293e-09,2.14578825819e-08,2.44392367676e-07,2.26393712666e-08,4.04212663341e-09,7.19016068906e-10,8.71130723112e-09,7.65549970083e-08,3.42661259367e-08,2.99158614547e-11,1.74995379298e-10,4.422831823e-10,2.23361912299e-10,4.74059369537e-09,1.15777509802e-09,2.84864931643e-09,1.60232105362e-09,5.49015375384e-08,5.3512899667e-10,1.39270099453e-07,1.03490435777e-07,2.57161104855e-06,8.52778914151e-09,4.72682313557e-06,2.0215715324e-07,1.55828232096e-07,3.727144815e-08,8.43976692977e-08,1.9180961317e-06,7.63014629257e-09,6.16340741999e-07,1.70064794247e-07,9.59979429282e-10,1.54982104927e-09,1.12265778895e-08,2.64978581299e-08,5.74687275545e-09,1.06492904772e-10,4.34249159298e-07,1.17178515779e-08,5.23592440516e-08,1.26180719207e-08,4.60859972407e-09,3.65175734096e-09,5.07973885178e-09,2.98924485165e-09,2.84083068181e-09,5.86193005248e-09,6.69928704156e-05,3.48765979652e-07,1.20125676517e-07,3.1504640674e-08,2.89324986191e-09,1.7602788116e-09,1.72403815668e-08,5.19891241169e-09,1.04490638364e-08,2.24569475904e-07,2.4755433814e-10,1.35320270545e-08,1.44215572817e-09,5.33786459478e-10,6.1780106364e-08,6.15033968199e-10,4.04212663341e-09,3.76126848778e-07,4.87803575311e-08,3.97074595426e-10,4.15670129428e-08,4.79348614135e-08,2.08083172915e-09,6.25436236135e-09,1.99114125188e-09,4.28317781243e-09,6.74884192975e-09,9.86066606146e-09,1.52716381208e-10,2.49224108018e-08,3.49482327522e-08,1.90776219711e-07,1.57680037915e-08,2.34463892745e-09,2.93599526913e-08,9.102408427e-08,2.83628427411e-08,2.23946017286e-09,2.4464515036e-07,3.60001592981e-08,4.68408245524e-07,2.92437192684e-07,1.09003863713e-08,3.81163260954e-08,5.43201883474e-09,0.000126401500893,5.2751261137e-09,3.25930429312e-08,5.53746559717e-09,1.96879601511e-09,3.00954261512e-09,5.85652970564e-10,8.27028934225e-09,9.16834252962e-08,3.73877178106e-08,1.46451088767e-06,2.21302798309e-07,1.34366860038e-06,3.45262329837e-09,1.23265920138e-07,1.18348919553e-09,1.92019777856e-09,1.27369421676e-08,1.52981840529e-08,2.55211801914e-07,6.40675139607e-08,1.92517336473e-06,1.13238263211e-08,2.36457822211e-06,1.44488637943e-06,8.7744533861e-10,1.14891696157e-08,7.23002528957e-07,6.09605588231e-10,1.0978276066e-08,3.18331245808e-07,8.74543637508e-09,5.8112443746e-09,4.33848263981e-08,4.53527260191e-09,1.2850900788e-08,5.4639088809e-09,8.74550298846e-09,3.36296665182e-07,7.64035412715e-09,2.16113660301e-09,1.45977683275e-08,7.79647706395e-08,1.21777446793e-07,9.1841023675e-10,7.96387311652e-09,3.26943485618e-08,2.16965823086e-08,2.94069877782e-07,2.23865725957e-09,1.05084985158e-09,6.93494817128e-09,3.32241456569e-09,2.02076955169e-09,3.6043492635e-09,9.3393461853e-09,2.78052109115e-10,2.48699535632e-07,4.21296526554e-08,8.28849699985e-09,1.89268369866e-08,1.58121551408e-08,1.34778630922e-07,4.70074064651e-06,9.6871675126e-09,1.22261001323e-08,9.36908861604e-09,6.89452450686e-09,9.98950699937e-09,2.51254861361e-09,2.14081918859e-10,4.91279834591e-08,5.2522281635e-08,3.07892461526e-10,4.11364524666e-08,7.61541329997e-10,2.47704292633e-08,4.13539513744e-09,5.31775690149e-09,1.75606764685e-08,6.11828738784e-08,2.18236961835e-09,3.38470265149e-08,4.38206093634e-08,1.21473320291e-07,1.56722217071e-06,1.66041695593e-06,6.64456933919e-08,1.24945183089e-08,1.87286808284e-08,7.01872915343e-09,1.86142479208e-09,3.22790447171e-06,1.31153896632e-08,3.18122914678e-08,2.9061100193e-08,2.94848567783e-09,3.45881256969e-09,2.79428732028e-05,1.35429814918e-07,1.03292929765e-09,2.79875820297e-08,1.04292674497e-09,2.90920461055e-08,1.18015930362e-10,1.79101977693e-09,5.59201289718e-08,3.8565133309e-07,3.18487778372e-08,1.24303289795e-06,4.65101335223e-09,2.98948563682e-08,1.1526358179e-07,1.27607413525e-09,1.45609231339e-08,1.36157547459e-08,4.52428272624e-09,4.98050933828e-09,1.61861102299e-08,1.64932689728e-08,5.75484193632e-07,2.88448154251e-10,8.72156480369e-09,5.04435831772e-07,9.74658576069e-09,8.14336402755e-08,1.03197528745e-07,1.5499897188e-07,1.60785393888e-08,7.00569614764e-07,1.99396215095e-08,2.11856132637e-08,3.26751314894e-09,2.20594364997e-09,1.78755149571e-10,1.31278465876e-09,2.15129372649e-10,1.38777354053e-08,1.14328969403e-07,1.43561633692e-08,4.23846664432e-08,1.75571179817e-09,2.01744998485e-09,1.57802482192e-08,2.06293004901e-09,9.82435466312e-10,1.64576874351e-09,4.48015313737e-09,7.58340945595e-10,2.02457739462e-09,3.45864314966e-08,1.02719207007e-06,2.91125861196e-08,5.51166723373e-10,9.29587129495e-10,2.81604837227e-08,3.5670011389e-09,8.60750688503e-08,7.20443082969e-08,8.2940999846e-08,6.8171746026e-10,1.83422372402e-07,8.06065429515e-07,4.36652562996e-08,2.66251333869e-07,3.35361801262e-05,3.02376754746e-08,9.57722567918e-10,7.94625698575e-09,3.57389295935e-09,5.24670584756e-10,7.90880392287e-07,1.74957759391e-09,1.67852964861e-09,3.35777947669e-08,3.41344708055e-09,4.82162310078e-10,8.41690734887e-08,2.13707611607e-08,8.98007570527e-08,1.33530064783e-09,6.24490947843e-09,9.29961385676e-10,5.56029167154e-10,9.87246639994e-09,5.65989166645e-09,3.11627879057e-08,2.66469672994e-08,4.93691798553e-09,1.89813453844e-09,2.21271658774e-08,1.12329612278e-08,3.36621042152e-09,1.66830318449e-07,3.66748881264e-10,2.6266663844e-10,5.42711313756e-06,1.12816744835e-09,2.2386660703e-07,9.98240157202e-09,1.04002690904e-08,2.97788091075e-07,2.21901395037e-08,1.19992336067e-08,2.9712878824e-09,1.40974067975e-08,1.92539548749e-08,1.81788098463e-10,2.46362263923e-09,4.06885405368e-08,3.20102273577e-08,2.9958544534e-09,6.6072053384e-10,1.49120555903e-08,6.81161793636e-08,4.17911337536e-07,5.3656884802e-07,1.83048847191e-08,4.20277812552e-09,3.17362608415e-10,2.64517802862e-06,2.76178138137e-08,1.87263946572e-08,4.63127314276e-09,1.24319043948e-08,3.75353366167e-08,5.47745855783e-07,4.052519742e-08,3.76785109779e-08,1.22151444515e-08,4.42744472195e-08,4.1541196083e-07,6.84092981373e-07,3.18850945646e-10,1.88774706977e-09,8.24023160817e-09,1.9161742415e-08,2.13367936652e-08,7.126824908e-09,1.38986893106e-08,6.24222138867e-06,7.04346305724e-08,9.74006724164e-08,4.19568380039e-08,5.06060804373e-10,1.19041976276e-08,2.58592991464e-09,5.19976808278e-08,4.74033812203e-10,7.43229788824e-09,6.39422292892e-09,8.58311182128e-08,1.40079037259e-09,4.5372562596e-08,5.34364053006e-10,1.44257725765e-08,8.0823152615e-09,1.03339887758e-08,4.70433025868e-09,7.43872661246e-08,2.8592259671e-09,7.7566959078e-08,3.15900230419e-08,2.5234052714e-07,2.5164060844e-06,3.63432036465e-07,1.15590516714e-07,5.24056753548e-09,1.29291584017e-06,6.59858514496e-08,2.99063085407e-09,2.30238885734e-08,5.19714493663e-10,3.42652768381e-08,6.86214551848e-09,9.23913567874e-09,3.40438344182e-09,1.10706492862e-08,4.60252991275e-09,1.52611434601e-08,2.7420352211e-08,1.5812698706e-08,2.28109726663e-08,1.91941875727e-08,1.5320216562e-06,1.1302491032e-07,1.11945253067e-08,1.27353083634e-09,7.11268821618e-09,1.30278774435e-08,4.7980547313e-10,3.9481090397e-08,2.85834661495e-10,1.23787140538e-08,3.86532065022e-07,1.36877513768e-08,5.93481104261e-08,3.97377108996e-09,2.04024708239e-08,0.000822657370009,4.76618396306e-07,1.49096868185e-09,4.21170920362e-09,2.36606911841e-08,6.85679468759e-09,3.96074248954e-08,2.24650307246e-08,1.47247520843e-08,2.24832263918e-10,1.61569317925e-08,7.20778103869e-09,5.01877046588e-08,6.27165652745e-10,2.27642047435e-08,4.47206147669e-08,2.90494495125e-09,1.08595210602e-09,4.02116668852e-08,1.93247053915e-08,1.18362981638e-08,7.6604322885e-09,1.00440292954e-05,4.42506369325e-08,1.5745042603e-08,1.56153934228e-08,6.31638630288e-10,2.30591350459e-08,1.00208163989e-09,1.94799856246e-08,8.61122995133e-09,1.13998694928e-08,3.09332577331e-08,8.11101301679e-08,1.26010197832e-08,9.24409349068e-08,1.57069720785e-07,1.02345677533e-06,1.09363513801e-08,6.98670760357e-07,1.30934125764e-07,1.15013928936e-10,9.43412814536e-09,1.18813889283e-10,4.84552897806e-10,3.36088379349e-10,3.57952956165e-08,4.95326113459e-10,5.49219407731e-07,6.23777012265e-08,3.06387901161e-11,1.74717995627e-09,5.73921621339e-09,6.82316851908e-07,3.26309734788e-09,2.49589106716e-07,2.16870121861e-09,5.42860112418e-09,1.03261421636e-09,6.16712085844e-08,3.67524233269e-09,1.11598188468e-08,5.84217957567e-07,2.69844576906e-06,7.46715311806e-09,3.2337539313e-09,5.01746644233e-09,1.0540702533e-08,1.65704907573e-08,1.98536165286e-09,7.23927717772e-09,3.2009801032e-08,3.72195674281e-09,3.95225097094e-09,4.46119123865e-08,1.13026857207e-07,1.02400890682e-05,4.63584228783e-07,6.75707223508e-09,6.544500053e-08,3.46664563722e-09,6.64416033302e-09,4.23038990505e-08,1.4173727747e-09,1.45972949284e-08,9.22586659846e-11,1.90851920934e-07,1.33827114723e-07,3.62016638888e-10,1.40611806643e-08,3.50719497888e-09,5.90325655025e-08,1.72524934783e-07,2.00772358738e-08,1.61480908645e-08,3.37841865594e-09,4.67592764508e-09,1.41100500173e-08,1.47254546334e-08,1.02148256076e-07,3.45949713321e-10,6.36814334598e-08,4.30433200194e-09,1.75641954314e-08,6.38345931669e-10,2.73295919229e-09,1.0094173275e-09,5.87554823142e-05,4.24223323137e-09,4.25335962007e-08,1.10411271237e-08,1.25616645974e-08,1.08003817001e-09,5.2807557438e-08,1.31323112385e-08,1.66743223673e-08,3.3072545591e-08,7.60330909344e-10,7.07803309297e-08,8.2484081787e-09,9.74082581706e-07,3.709883174e-07,1.50470029325e-07,5.28756638474e-09,2.22771490144e-09,2.07282591091e-09,6.60395405028e-10,5.58114408022e-08,5.93364539725e-08,6.89217083405e-09,1.74010272858e-09,5.3538934397e-10,7.02640434724e-09,1.85472670555e-09,8.50644354955e-09,1.07743183264e-08,4.48011462595e-07,6.76795564036e-10,2.09540851337e-09,6.42826680775e-09,2.60290953236e-08,1.49798523807e-07,1.36791644678e-08,2.80590981561e-08,6.36515622432e-08,1.06070556782e-08,1.0616219015e-09,1.30512560759e-08,1.43736986757e-08,3.31528027253e-09,2.49821869858e-08,2.24317941999e-09,2.60343802072e-10,2.42276443352e-09,1.11456571972e-07,2.78231162554e-08,3.25123772349e-09,3.4850229369e-09,3.28766147639e-09,9.13343427555e-08,5.19394416365e-10,1.14166789444e-06,3.59804386285e-09,1.47500585967e-07,1.69855276511e-08,8.49023820138e-08,5.09642426039e-08,2.16740811965e-08,2.89150081656e-09,2.45370412877e-09,1.30757751293e-08,1.10571374279e-09,8.04842503754e-09,1.41476164117e-08,2.85958057233e-09,1.0764751579e-07,3.97466504154e-09,1.97512228794e-09,2.28980368888e-07,1.00418304783e-08,4.75505821385e-08,8.44433998282e-08,1.0996655142e-08,7.82303199998e-09,5.70849456594e-09,1.68631544284e-08,1.03074926372e-08,4.48380943485e-10,3.46633015624e-08,1.71976850538e-08,1.9569064591e-08,8.64471438877e-10,1.33551156978e-06,3.58565905856e-08,6.4385248244e-09,6.2455320915e-08,2.25682015298e-07,3.58684082435e-09,2.54916983522e-07,6.1568485421e-08,9.94416216038e-10,1.1439396097e-09,1.41783868912e-06,4.45174741515e-08,5.48658896093e-10,7.12060144181e-09,2.12923172427e-08,4.10190850175e-08,1.22808359038e-08,1.04305248882e-08,9.77264580371e-09,1.35712818761e-09,6.04646501756e-08,2.55852938835e-09,3.1001048395e-09,2.01484251505e-09,1.07902566882e-08,1.5799235531e-10,1.05731269286e-05,1.49632253255e-08,4.58799798153e-09,9.75695968464e-10,1.1291926505e-08,1.33848809813e-09,2.1543385742e-08,9.2730279011e-10,2.22043663456e-08,2.16700590805e-09,6.1971148213e-09,2.69924971263e-09,6.42157900188e-08,1.12571811428e-07,3.76991904361e-09,4.11004341672e-09,3.37328849298e-08,2.38760144988e-09,2.2498694463e-08,1.30288901889e-09,3.2620082191e-09,1.29693987105e-07,3.22870619129e-09,2.00265512831e-05,2.07431694044e-09,2.70306754757e-09,1.34632367477e-08,1.81848971437e-08,4.42977388104e-09,8.59603872527e-08,1.07743183264e-08,1.05260014038e-07,2.79118683721e-09,9.16790021677e-09,3.44979667943e-07,2.6413829346e-07,1.63291868915e-08,1.35945530388e-08,2.19136995838e-06,8.01184563137e-10,7.18430315239e-11,2.48769254085e-07,5.56280070896e-08,5.65548852194e-09,2.94089219643e-09,8.31219804098e-09,5.91503157565e-09,1.21031695777e-09,2.62988075761e-09,2.73439439979e-08,2.5637625356e-08,1.33169981709e-08,2.1910127046e-08,2.0447467719e-08,8.44048653192e-09,4.28620410275e-08,7.68906160964e-10,2.70476779862e-10,8.46153813683e-09,2.45799958165e-09,2.76714762215e-09,3.47556361469e-09,1.05749421664e-07,3.61393368564e-08,8.86774387254e-07,1.21789511809e-07,2.71214117831e-09,2.80949254972e-08,1.28289883605e-07,2.02265892923e-09,1.33133593039e-09,1.52779400242e-09,5.27491721414e-08,8.35952529421e-10,8.6292164525e-09,4.94302332399e-09,7.26112592275e-09,7.4252881177e-08,8.39865332836e-09,5.85470072423e-09,6.87853907166e-09,6.11436457021e-09,1.25369453485e-07,3.7539021891e-10,7.59049001431e-09,9.32761921035e-08,1.8258404566e-08,3.83290377215e-09,1.09292201955e-08,2.60031773891e-09,3.60854031101e-08,9.35244884204e-07,2.62420371655e-07,1.27369603753e-09,1.35805588997e-08,3.21989968022e-09,1.48700107783e-09,2.88663515313e-09,1.94047427016e-08,2.65608424144e-09,6.21181612814e-08,2.55060195187e-09,6.61879084873e-08,6.94394486356e-10,3.09621661643e-09,3.27893628915e-10,1.25911698845e-08,2.02139679994e-10,1.88886204455e-07,4.24246582309e-10,8.00982891125e-10,1.94374460989e-05,1.02202193375e-08,1.05240482995e-08,1.32518479745e-07,5.08174800018e-08,5.43706413225e-10,6.33639585246e-09,1.20849055207e-08,0.000268270378001,3.65410031122e-08,2.44668605376e-08,1.34596671586e-08,2.44277842398e-08,1.21647081297e-09,1.62706981222e-09,6.60451533463e-08,1.43299205835e-09,5.26098347109e-07,4.23513114356e-07,1.05324033939e-08,5.22202356024e-07,9.51724365983e-10,5.65501379057e-09,5.57084645081e-09,2.81869105834e-09,5.79126124833e-09,2.27867502645e-08,5.31037791518e-09,3.37976557852e-09,5.2974819198e-08,3.5479718008e-07,1.26374716247e-10,1.46820413605e-08,3.00980957491e-07,2.64253174009e-07,2.49447147382e-08,2.76429390489e-09,7.72481545397e-09,6.07474825998e-09,1.42696245931e-08,5.18831555496e-08,3.42654722374e-08,7.96458721197e-09,2.82194694279e-08,2.86243861947e-10,6.71918520823e-09,4.94763838788e-08,4.15650891483e-09,1.31525084157e-07,1.92558191614e-09,8.2485973607e-09,2.18488231951e-08,8.94681839725e-09,9.53746507548e-07,3.75051385504e-08,0.998324215412,4.12981080444e-07,6.62065394863e-06,7.5283786316e-08,4.80215023302e-09,1.32644224493e-08,2.62651496996e-07,1.73241065937e-08,1.37196920491e-08,1.2738392563e-09,2.54739784822e-09,1.86345672226e-09,1.50746956251e-07,4.60499727239e-09,5.90644674503e-07,5.63869662074e-09,6.87168437707e-08,6.56505426377e-06,5.33374802103e-08,1.17978462555e-08,6.92124491053e-09,2.21193907635e-09,8.41824721043e-10,2.05577848078e-08,7.71870301008e-09,2.75015765716e-09,5.31665413916e-08,1.17122658239e-08,1.80792092408e-08,3.04511829086e-08,7.32757365896e-09,1.50960230427e-10,2.46080000821e-09,1.01204538083e-08,1.17485559059e-08,3.85165276384e-06,8.79389716602e-08,2.86378343262e-09,1.52274139964e-08,8.71557381821e-10,2.1586535226e-08,9.6446850506e-10,2.36894903693e-09,2.33654939841e-09,4.22235295616e-08,1.58168833586e-09,2.6060242817e-06,5.47760670155e-10,4.22205523876e-08,8.94596574597e-09,1.22308110306e-08,4.92667631136e-08,2.65706354696e-08,1.09781712609e-08,2.22291030028e-09,7.61132223914e-09,7.10765279965e-10,1.03685245278e-07,5.13748066311e-09,2.02421435169e-09,2.51700580378e-08,2.06493314181e-06,5.17693532487e-09,1.90773008502e-10,1.03199749901e-06,3.10153591698e-09,9.77318670436e-09,3.82069975657e-09,1.7080971304e-07,8.44389669297e-10,2.96870888983e-10,1.50744139393e-10,2.9359503273e-08,3.86211107539e-09,5.21274401422e-09,4.51571313675e-09,1.06739992134e-06,2.24823502037e-08,2.16627160654e-08,2.95789952531e-08,2.25851026769e-09,7.43435748518e-08,6.9356755894e-09,1.43442031586e-08,3.00778646434e-09,5.835658623e-09,1.46132013157e-08,1.53119330548e-08,5.54732881852e-09,4.39625713611e-09,2.1968668662e-08,1.2611768625e-06,4.74138950324e-09,3.44012568432e-10,5.38340003686e-08,4.96851393361e-10,2.41838304937e-09,7.15727077605e-09,1.28520705189e-09,2.85178794002e-06,2.45969857815e-08,1.00318375829e-08,2.32568861946e-08,6.13501001112e-08,1.24228973775e-08,2.05728253877e-10,2.24473751587e-07,2.54266154798e-08,4.89456652986e-09,8.42858582928e-08]
for i in range(len(predict)):
  a.v[i] = predict[i]
print(predict)

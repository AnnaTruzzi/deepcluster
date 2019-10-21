import pickle
import os
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

## define images names and synsets
img_names_cichy118 = {'001':'orange','002':'bench','003':'remote','004':'car','005':'stove','006':'man','007':'table','008':'apple','009':'wagon','010':'dog','011':'fox','012':'bus','013':'train','014':'ipod',
                '015':'pizza','016':'bird','017':'horse','018':'laptop','019':'bear','020':'basketball ball','021':'piano','022':'guitar','023':'baseball ball','024':'seal','025':'chair','026':'orangutan','027':'bowl',
                '028':'tiger','029':'moped','030':'tie','031':'printer','032':'lion','033':'neil','034':'drum','035':'bow','036':'fig','037':'butterfly','038':'lamp','039':'banana','040':'sofa',
                '041':'lemon','042':'hoover','043':'hammer','044':'sunglasses','045':'bike','046':'rabbit','047':'beaker','048':'elephant','049':'microwave','050':'volleyball ball','051':'strawberry',
                '052':'sheep','053':'frog','054':'washing machine','055':'fridge','056':'turtle','057':'ax','058':'helmet','059':'camel','060':'lipstick','061':'airplane','062':'dishwasher','063':'burger',
                '064':'backpack','065':'purse','066':'hamster','067':'microphone','068':'mushroom','069':'cow','070':'violin','071':'orca','072':'cucumber','073':'thermometer','074':'pineapple','075':'harp',
                '076':'squirrel','077':'notebook','078':'trumpet','079':'plastic bag','080':'jug','081':'snail','082':'toaster','083':'banjo','084':'screwdriver','085':'snowmobile','086':'pig',
                '087':'pomegranate','088':'accordion','089':'racket','090':'bagel','091':'trombone','092':'syringe','093':'fish','094':'hair_dryer','095':'starfish','096':'hotdog','097':'ladybug',
                '098':'aircraft carrier','099':'jellyfish','100':'coffee maker','101':'television','102':'pretzel','103':'artichoke','104':'golf ball','105':'alarm clock','106':'traffic light',
                '107':'bottle','108':'pan','109':'weights','110':'football ball','111':'sweet pepper','112':'corkscrew','113':'tennis ball','114':'computer mouse','115':'mug','116':'keyboard',
                '117':'can opener','118':'fan'}

img_synsets_cichy118 = {'001':'orange.n.01','002':'bench.n.01','003':'remote_control.n.01','004':'car.n.01','005':'stove.n.01','006':'man.n.03','007':'table.n.02','008':'apple.n.01','009':'cart.n.01','010':'dog.n.01',
                '011':'fox.n.01','012':'bus.n.01','013':'train.n.01','014':'ipod.n.01','015':'pizza.n.01','016':'bird.n.01','017':'horse.n.01','018':'laptop.n.01','019':'bear.n.01','020':'basketball.n.02',
                '021':'piano.n.01','022':'guitar.n.01','023':'baseball.n.02','024':'seal.n.09','025':'chair.n.01','026':'orangutan.n.01','027':'bowl.n.01','028':'tiger.n.02','029':'moped.n.01','030':'necktie.n.01',
                '031':'printer.n.03','032':'lion.n.01','033':'nail.n.02','034':'drum.n.01','035':'bow.n.04','036':'fig.n.04','037':'butterfly.n.01','038':'lamp.n.02','039':'banana.n.02',
                '040':'sofa.n.01','041':'lemon.n.01','042':'vacuum.n.04','043':'hammer.n.02','044':'sunglasses.n.01','045':'bicycle.n.01','046':'rabbit.n.01','047':'beaker.n.01','048':'elephant.n.01',
                '049':'microwave.n.02','050':'volleyball.n.02','051':'strawberry.n.01','052':'sheep.n.01','053':'frog.n.01','054':'washer.n.03','055':'electric_refrigerator.n.01','056':'turtle.n.02','057':'ax.n.01',
                '058':'helmet.n.02','059':'camel.n.01','060':'lipstick.n.01','061':'airplane.n.01','062':'dishwasher.n.01','063':'hamburger.n.01','064':'backpack.n.01','065':'bag.n.04','066':'hamster.n.01',
                '067':'microphone.n.01','068':'mushroom.n.05','069':'cow.n.02','070':'violin.n.01','071':'killer_whale.n.01','072':'cucumber.n.02','073':'thermometer.n.01','074':'pineapple.n.02','075':'harp.n.01',
                '076':'squirrel.n.01','077':'notebook.n.01','078':'cornet.n.01','079':'plastic_bag.n.01','080':'pitcher.n.02','081':'snail.n.01','082':'toaster.n.02','083':'banjo.n.01',
                '084':'screwdriver.n.01','085':'snowmobile.n.01','086':'hog.n.03','087':'pomegranate.n.02','088':'accordion.n.01','089':'racket.n.04','090':'bagel.n.01','091':'trombone.n.01',
                '092':'syringe.n.01','093':'fish.n.01','094':'hand_blower.n.01','095':'starfish.n.01','096':'frank.n.02','097':'ladybug.n.01','098':'aircraft_carrier.n.01','099':'jellyfish.n.02',
                '100':'coffee_maker.n.01','101':'television_receiver.n.01','102':'pretzel.n.01','103':'artichoke.n.02','104':'golf_ball.n.01','105':'alarm_clock.n.01','106':'traffic_light.n.01',
                '107':'bottle.n.01','108':'pan.n.01','109':'weight.n.02','110':'ball.n.01','111':'sweet_pepper.n.02','112':'corkscrew.n.01','113':'tennis_ball.n.01','114':'mouse.n.04','115':'mug.n.04',
                '116':'keyboard.n.01','117':'can_opener.n.01','118':'fan.n.01'}

img_w2v_cichy118 = {'001':'orange','002':'bench','003':'remote','004':'car','005':'stove','006':'man','007':'table','008':'apple','009':'wagon','010':'dog','011':'fox','012':'bus','013':'train','014':'ipod',
                '015':'pizza','016':'bird','017':'horse','018':'laptop','019':'bear','020':'basketball ball','021':'piano','022':'guitar','023':'baseball ball','024':'seal','025':'chair','026':'orangutan','027':'bowl',
                '028':'tiger','029':'moped','030':'tie','031':'printer','032':'lion','033':'neil','034':'drum','035':'bow','036':'fig','037':'butterfly','038':'lamp','039':'banana','040':'sofa',
                '041':'lemon','042':'hoover','043':'hammer','044':'sunglasses','045':'bike','046':'rabbit','047':'beaker','048':'elephant','049':'microwave','050':'volleyball ball','051':'strawberry',
                '052':'sheep','053':'frog','054':'washing machine','055':'fridge','056':'turtle','057':'ax','058':'helmet','059':'camel','060':'lipstick','061':'airplane','062':'dishwasher','063':'burger',
                '064':'backpack','065':'purse','066':'hamster','067':'microphone','068':'mushroom','069':'cow','070':'violin','071':'orca','072':'cucumber','073':'thermometer','074':'pineapple','075':'harp',
                '076':'squirrel','077':'notebook','078':'trumpet','079':'plastic bag','080':'jug','081':'snail','082':'toaster','083':'banjo','084':'screwdriver','085':'snowmobile','086':'pig',
                '087':'pomegranate','088':'accordion','089':'racket','090':'bagel','091':'trombone','092':'syringe','093':'fish','094':'hair_dryer','095':'starfish','096':'hotdog','097':'ladybug',
                '098':'aircraft carrier','099':'jellyfish','100':'coffee maker','101':'television','102':'pretzel','103':'artichoke','104':'golf ball','105':'alarm clock','106':'traffic light',
                '107':'bottle','108':'pan','109':'weights','110':'football ball','111':'sweet pepper','112':'corkscrew','113':'tennis ball','114':'computer mouse','115':'mug','116':'keyboard',
                '117':'can opener','118':'fan'}


img_names_niko92 = {'01':'hand_1','02':'ear','03':'hand_2','04':'man_1','05':'hair','06':'woman_1','07':'woman_2','08':'hand_3','09':'eye','10':'man_2','11':'hand_4','12':'fist','13':'human face_1','14':'human face_2',
                '15':'human face_3','16':'human face_4','17':'human face_5','18':'human face_6','19':'human face_7','20':'human face_8','21':'human face_9','22':'human face_10','23':'human face_11','24':'human face_12',
                '25':'armadillo','26':'camel','27':'snake_1','28':'wolf','29':'monkey_1','30':'ostrich','31':'snake_2','32':'zebra','33':'elephant','34':'monkey_2','35':'sheep','36':'frog','37':'cow',
                '38':'goat','39':'monkey face_1','40':'monkey face_2','41':'dog_1','42':'dog_2','43':'monkey face_3','44':'monkey face_4','45':'lizard','46':'giraffe','47':'lion','48':'monkey_3','49':'carrot',
                '50':'grape','51':'potato','52':'bush','53':'jalapeno','54':'lettuce','55':'kiwi','56':'zucchini','57':'leaf','58':'apple','59':'radish','60':'eggplant','61':'lake','62':'pine cone',
                '63':'banana','64':'tomato','65':'garlic','66':'hayfield','67':'tree','68':'pineapple','69':'pear','70':'sweet pepper','71':'waterfall','72':'city','73':'bottle','74':'light bulb',
                '75':'roundabout sign','76':'musicassette','77':'belfry','78':'flag','79':'key','80':'pliers','81':'monument','82':'door','83':'hammer','84':'chair','85':'gun','86':'house_1',
                '87':'dome','88':'umbrella','89':'mobile phone','90':'house_2','91':'stove','92':'road sign'}

img_synsets_niko92 = {'01':'hand.n.01','02':'ear.n.03','03':'hand.n.01','04':'man.n.01','05':'hair.n.01','06':'woman.n.01','07':'woman.n.01','08':'hand.n.01','09':'eye.n.01','10':'man.n.01','11':'hand.n.01',
                '12':'fist.n.01','13':'face.n.01','14':'face.n.01','15':'face.n.01','16':'face.n.01','17':'face.n.01','18':'face.n.01','19':'face.n.01','20':'face.n.01','21':'face.n.01','22':'face.n.01',
                '23':'face.n.01','24':'face.n.01','25':'armadillo.n.01','26':'camel.n.01','27':'snake.n.01','28':'wolf.n.01','29':'monkey.n.01','30':'ostrich.n.02','31':'snake.n.01','32':'zebra.n.01',
                '33':'elephant.n.01','34':'monkey.n.01','35':'sheep.n.01','36':'frog.n.01','37':'cow.n.01','38':'goat.n.01','39':'monkey.n.01','40':'monkey.n.01','41':'dog.n.01','42':'dog.n.01','43':'monkey.n.01',
                '44':'monkey.n.01','45':'lizard.n.01','46':'giraffe.n.01','47':'lion.n.01','48':'monkey.n.01','49':'carrot.n.01','50':'grape.n.01','51':'potato.n.01','52':'shrub.n.01','53':'jalapeno.n.02',
                '54':'lettuce.n.02','55':'chinese_gooseberry.n.01','56':'zucchini.n.01','57':'leaf.n.01','58':'apple.n.01','59':'radish.n.01','60':'eggplant.n.01','61':'lake.n.01','62':'pinecone.n.01',
                '63':'banana.n.02','64':'tomato.n.01','65':'garlic.n.01','66':'hayfield.n.01','67':'tree.n.01','68':'pineapple.n.01','69':'pear.n.01','70':'sweet_pepper.n.02','71':'waterfall.n.01','72':'city.n.01',
                '73':'bottle.n.01','74':'lightbulb.n.01','75':'signpost.n.01','76':'cassette.n.01','77':'belfry.n.01','78':'flag.n.01','79':'key.n.01','80':'pliers.n.01','81':'memorial.n.03','82':'door.n.01',
                '83':'hammer.n.02','84':'chair.n.01','85':'gun.n.01','86':'house.n.01','87':'dome.n.01','88':'umbrella.n.01','89':'cellphone.n.01','90':'house.n.01','91':'stove.n.01','92':'signpost.n.01'}

img_w2v_niko92 = {'01':'hand','02':'ear','03':'hand','04':'man','05':'hair','06':'woman','07':'woman','08':'hand','09':'eye','10':'man','11':'hand','12':'fist','13':'human face','14':'human face',
                '15':'human face','16':'human face','17':'human face','18':'human face','19':'human face','20':'human face','21':'human face','22':'human face','23':'human face','24':'human face',
                '25':'armadillo','26':'camel','27':'snake','28':'wolf','29':'monkey','30':'ostrich','31':'snake','32':'zebra','33':'elephant','34':'monkey','35':'sheep','36':'frog','37':'cow',
                '38':'goat','39':'monkey face','40':'monkey face','41':'dog','42':'dog','43':'monkey face','44':'monkey face','45':'lizard','46':'giraffe','47':'lion','48':'monkey','49':'carrot',
                '50':'grape','51':'potato','52':'bush','53':'jalapeno','54':'lettuce','55':'kiwi','56':'zucchini','57':'leaf','58':'apple','59':'radish','60':'eggplant','61':'lake','62':'pine cone',
                '63':'banana','64':'tomato','65':'garlic','66':'hayfield','67':'tree','68':'pineapple','69':'pear','70':'sweet pepper','71':'waterfall','72':'city','73':'bottle','74':'light bulb',
                '75':'roundabout sign','76':'musicassette','77':'belfry','78':'flag','79':'key','80':'pliers','81':'monument','82':'door','83':'hammer','84':'chair','85':'gun','86':'house',
                '87':'dome','88':'umbrella','89':'mobile phone','90':'house','91':'stove','92':'road sign'}



with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_names.pickle', 'wb') as handle:
    pickle.dump(img_names_cichy118, handle)
with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_synsets.pickle', 'wb') as handle:
    pickle.dump(img_synsets_cichy118, handle)
with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_w2v.pickle', 'wb') as handle:
    pickle.dump(img_w2v_cichy118, handle)

with open('/home/CUSACKLAB/annatruzzi/cichy2016/niko92_img_names.pickle', 'wb') as handle:
    pickle.dump(img_names_niko92, handle)
with open('/home/CUSACKLAB/annatruzzi/cichy2016/niko92_img_synsets.pickle', 'wb') as handle:
    pickle.dump(img_synsets_niko92, handle)
with open('/home/CUSACKLAB/annatruzzi/cichy2016/niko92_img_w2v.pickle', 'wb') as handle:
    pickle.dump(img_w2v_niko92, handle)
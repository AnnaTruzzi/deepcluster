import pickle
import os

## define images names and synsets
img_names = {'001':'orange','002':'bench','003':'remote','004':'car','005':'stove','006':'man','007':'table','008':'apple','009':'wagon','010':'dog','011':'fox','012':'bus','013':'train','014':'ipod',
                '015':'pizza','016':'bird','017':'horse','018':'laptop','019':'bear','020':'basketball','021':'piano','022':'guitar','023':'baseball','024':'seal','025':'chair','026':'orangutan','027':'bowl',
                '028':'tiger','029':'moped','030':'tie','031':'printer','032':'lion','033':'neil','034':'drum','035':'bow','036':'fig','037':'butterfly','038':'lamp','039':'banana','040':'sofa',
                '041':'lemon','042':'hoover','043':'hammer','044':'sunglasses','045':'bike','046':'rabbit','047':'measuring jug','048':'elephant','049':'microwave','050':'volleyball','051':'strawberry',
                '052':'sheep','053':'frog','054':'washing machine','055':'fridge','056':'turtle','057':'axe','058':'helmet','059':'camel','060':'lipstick','061':'airplane','062':'dishwasher','063':'burger',
                '064':'backpack','065':'purse','066':'hamster','067':'microphone','068':'mushroom','069':'cow','070':'violin','071':'orca','072':'cucumber','073':'termometer','074':'pineapple','075':'harp',
                '076':'squirrel','077':'notebook','078':'trumpet','079':'plastic bag','080':'jug','081':'snail','082':'toaster','083':'benjo','084':'screwdriver','085':'snowbike','086':'pig',
                '087':'pomegranate','088':'accordion','089':'tennis racket','090':'bagel','091':'trombone','092':'syringe','093':'fish','094':'hair drier','095':'starfish','096':'hot dog','097':'ladybug',
                '098':'aircraft carrier','099':'jellyfish','100':'coffee machine','101':'television','102':'pretzel','103':'artichoke','104':'golf ball','105':'alarm clock','106':'traffic light',
                '107':'bottle','108':'pan','109':'weights','110':'football ball','111':'bellpepper','112':'corkscrew','113':'tennis ball','114':'computer mouse','115':'mug','116':'keyboard',
                '117':'can opener','118':'fan'}

img_synsets = {'001':'orange.n.01','002':'bench.n.01','003':'remote_control.n.01','004':'car.n.01','005':'stove.n.01','006':'man.n.03','007':'table.n.02','008':'apple.n.01','009':'cart.n.01','010':'dog.n.01',
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


with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_names.pickle', 'wb') as handle:
    pickle.dump(img_names, handle)
with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_synsets.pickle', 'wb') as handle:
    pickle.dump(img_synsets, handle)
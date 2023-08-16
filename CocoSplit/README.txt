HOW TO USE COCOSPLIT
-----------------------
1. Go to roboflow and zip up dataset (100% training, no augments, Format: COCO)
2. Place zip file into DATA folder of CocoSplit (Will need to scp onto Lambda first)
	-
3. Do: python start.py
4. Enter in the correct values (CASE SENSITIVE)
5. After using CocoSplit it will generate a seed, copy this and use it in a future run
   to get the same splits [on the same data]

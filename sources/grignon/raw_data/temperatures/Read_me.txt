# --- Donn�es de temp�rature de couvert et d'organe de bl�
# --- Campagne d'acquisition 2011-2012, INRA ECOSYS Grignon
# --- contact: michael.chelle@inra.fr

# Donn�es thermiques m�so/micro/phylloclimatiques acquises du 30/11/2011 au 23/05/2012

----------------------------------------------------------------------------------------------------------------------------------------------------
# File 1: "campagne_2011-2012_temperatures_mesoclimatiques.csv"
# Donn�es m�soclimatiques provenant d'une station m�t�o (mod�le Enerco 516i, CIMEL Electronique) � moins d'1 km de distance du site exp�rimental

Year = Ann�e / Month = Mois / Day = Jour / Hour = Heure
Temperature = Temp�rature d�air mesur�e � 2 m au-dessus du sol (mesure instantan�e horaire) en degr�s celsius
----------------------------------------------------------------------------------------------------------------------------------------------------
# File 2: "campagne_2011-2012_temperatures_microclimatiques.csv"
# Donn�es microclimatiques au sein du couvert de bl� 
# Temp�rature d'air mesur�e � l'aide de thermocouples fins de type T (0,06 mm de diam�tre)

Time = Jour x heure
Captor = indentifiant du thermocouple
Temperature = Temp�rature d'air moyenn�e sur l'heure
Height = Hauteur au-dessus du sol (0/25/50/75/100 cm au-dessus du sol)
ITK = itin�raire technique de la parcelle (conduite extensive = faible densit� de semis - 180 grains /m� - et apports azot�s - 65 kg/ha ; conduite intensive = forte densit� de semis - 250 grains / m� - et apports azot�s - 210 kg/ha)
----------------------------------------------------------------------------------------------------------------------------------------------------
# File 3: "campagne_2011-2012_temperatures_phylloclimatiques.csv"
# Donn�es phylloclimatiques au sein du couvert de bl� 
# Temp�rature des 3 limbes les plus jeunes du ma�tre-brin de 9 plantes de bl� par couvert	
# Thermocouples fins de type T (0,2 mm de diam�tre) positionn�s sous et en contact avec la face abaxiale

Captor_ID = indentifiant du thermocouple
ITK = itin�raire technique de la parcelle (conduite extensive = faible densit� de semis - 180 grains /m� - et apports azot�s - 65 kg/ha ; conduite intensive = forte densit� de semis - 250 grains / m� - et apports azot�s - 210 kg/ha)
DAY = Jour / HOUR = Heure
Leaf_level = �tage foliaire de la feuille sur laquelle le thermocouple a �t� positionn� (attention pour la modalit� "5.5" incertitude quant � l'�tage foliaire: 5 ou 6)
Temperature = Temp�rature de feuille moyenn�e sur l'heure
Plant = identifiant de la plante suivie (9 r�p�titions techniques par ITK i.e. par parcelle)
----------------------------------------------------------------------------------------------------------------------------------------------------
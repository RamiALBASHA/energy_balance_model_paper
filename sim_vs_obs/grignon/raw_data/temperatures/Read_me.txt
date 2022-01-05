# --- Données de température de couvert et d'organe de blé
# --- Campagne d'acquisition 2011-2012, INRA ECOSYS Grignon
# --- contact: michael.chelle@inra.fr

# Données thermiques méso/micro/phylloclimatiques acquises du 30/11/2011 au 23/05/2012

----------------------------------------------------------------------------------------------------------------------------------------------------
# File 1: "campagne_2011-2012_temperatures_mesoclimatiques.csv"
# Données mésoclimatiques provenant d'une station météo (modèle Enerco 516i, CIMEL Electronique) à moins d'1 km de distance du site expérimental

Year = Année / Month = Mois / Day = Jour / Hour = Heure
Temperature = Température d’air mesurée à 2 m au-dessus du sol (mesure instantanée horaire) en degrés celsius
----------------------------------------------------------------------------------------------------------------------------------------------------
# File 2: "campagne_2011-2012_temperatures_microclimatiques.csv"
# Données microclimatiques au sein du couvert de blé 
# Température d'air mesurée à l'aide de thermocouples fins de type T (0,06 mm de diamètre)

Time = Jour x heure
Captor = indentifiant du thermocouple
Temperature = Température d'air moyennée sur l'heure
Height = Hauteur au-dessus du sol (0/25/50/75/100 cm au-dessus du sol)
ITK = itinéraire technique de la parcelle (conduite extensive = faible densité de semis - 180 grains /m² - et apports azotés - 65 kg/ha ; conduite intensive = forte densité de semis - 250 grains / m² - et apports azotés - 210 kg/ha)
----------------------------------------------------------------------------------------------------------------------------------------------------
# File 3: "campagne_2011-2012_temperatures_phylloclimatiques.csv"
# Données phylloclimatiques au sein du couvert de blé 
# Température des 3 limbes les plus jeunes du maître-brin de 9 plantes de blé par couvert	
# Thermocouples fins de type T (0,2 mm de diamètre) positionnés sous et en contact avec la face abaxiale

Captor_ID = indentifiant du thermocouple
ITK = itinéraire technique de la parcelle (conduite extensive = faible densité de semis - 180 grains /m² - et apports azotés - 65 kg/ha ; conduite intensive = forte densité de semis - 250 grains / m² - et apports azotés - 210 kg/ha)
DAY = Jour / HOUR = Heure
Leaf_level = étage foliaire de la feuille sur laquelle le thermocouple a été positionné (attention pour la modalité "5.5" incertitude quant à l'étage foliaire: 5 ou 6)
Temperature = Température de feuille moyennée sur l'heure
Plant = identifiant de la plante suivie (9 répétitions techniques par ITK i.e. par parcelle)
----------------------------------------------------------------------------------------------------------------------------------------------------
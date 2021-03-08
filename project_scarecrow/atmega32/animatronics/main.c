/*
 * project_scarecrow_animatronics.c
 *
 * Created: 02/01/2020 07:21:10 PM
 * Author : VictorT
 */ 

#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <stdio.h>


#include "avr.h"

#define SET_BIT(p,i) ((p) |=  (1 << (i)))
#define CLR_BIT(p,i) ((p) &= ~(1 << (i)))
#define GET_BIT(p,i) ((p) &   (1 << (i)))


int main(void)
{
	const int enableLight1Pin = 1;
	const int enableLight2Pin = 2;

	const int light1Pin = 1;
	const int light2Pin = 2;

	bool light1 = false;
	bool light2 = false;

	const int maxLightCoutner = 10000;
	const int midLightCounter = 5000;
	const int overheadDelay = -5000;
	int lightCounter = 0;
	
	DDRA = 0b00111111;
	DDRC = 0b00000001;
	DDRD = 0b00000000;
	
    while (1) 
    {
		light1 = false;
		light2 = false;

		if (GET_BIT(PIND, enableLight1Pin)) {
			light1 = true;
		} else if (GET_BIT(PIND, enableLight2Pin)) {
			light2 = true;
		}
		
		if (light1) {
			if (lightCounter < 0) {
				lightCounter = lightCounter + 1;
			} else if (lightCounter > 0 && lightCounter < maxLightCounter) {
				if (lightCounter < midLightCounter) {
					SET_BIT(PORTA, light1Pin);
					avr_wait(2);
				} else {
					SET_BIT(PORTA, light1Pin);
					avr_wait(1);
					CLR_BIT(PORTA, light1Pin);
					avr_wait(1);
				}
				lightCounter = lightCounter + 1;
			} else {
				lightCounter = overheadDelay;
			}

		} else if (light2) {
			if (lightCounter < 0) {
				lightCounter = lightCounter + 1;
			} else if (lightCounter > 0 && lightCounter < maxLightCounter) {
				if (lightCounter < midLightCounter) {
					SET_BIT(PORTA, light2Pin);
					avr_wait(2);
				} else {
					SET_BIT(PORTA, light2Pin);
					avr_wait(1);
					CLR_BIT(PORTA, light2Pin);
					avr_wait(1);
				}
				lightCounter = lightCounter + 1;
			} else {
				lightCounter = overheadDelay;
			}
		} else {
			lightCounter = lightCounter - 1;		
		}
    }
}


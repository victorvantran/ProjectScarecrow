/*
 * project_scarecrow_stepper_motor.c
 *
 * Created: 01/8/2020 09:56:16 PM
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

const int ms1Pin = 2;
const int ms2Pin = 3;
const int ms3Pin = 5;

const int pwmPin = 0;
const int directionPin = 2;


const int enableIn = 0
const int directionIn = 1;


int main(void)
{
	DDRA = 0b00111111;
	DDRC = 0b00000001;
	DDRD = 0b00000000;
    while (1) 
    {	
		if (GET_BIT(PIND, directionIn)) {
			SET_BIT(PORTA, directionPin);
		} else {
			CLR_BIT(PORTA, directionPin);
		}
			
		int step = 1;

		if (!GET_BIT(PIND, ms1Pin) && !GET_BIT(PIND, ms2Pin) && !GET_BIT(PIND, ms3Pin)) {
			step = 1;
		} else if (GET_BIT(PIND, ms1Pin) && !GET_BIT(PIND, ms2Pin) && !GET_BIT(PIND, ms3Pin)) {
			step = 2;
		} else if (!GET_BIT(PIND, ms1Pin) && GET_BIT(PIND, ms2Pin) && !GET_BIT(PIND, ms3Pin)) {
			step = 4;
		} else if (GET_BIT(PIND, ms1Pin) && GET_BIT(PIND, ms2Pin) && !GET_BIT(PIND, ms3Pin)) {
			step = 8;
		} else if (GET_BIT(PIND, ms1Pin) && GET_BIT(PIND, ms2Pin) && GET_BIT(PIND, ms3Pin)) {
			step = 16;
		}
		
				
		// PWM
		if (GET_BIT(PIND, enableIn)) {
			SET_BIT(PORTA, pwmPin);
			pwm_wait(step);
			CLR_BIT(PORTA, pwmPin);
			pwm_wait(step);
		} else {
			pwm_wait(1);
		}
		
    }
}


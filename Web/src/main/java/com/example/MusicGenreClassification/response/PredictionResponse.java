package com.example.MusicGenreClassification.response;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class PredictionResponse {
    private String genre;
    private double confidence;

}

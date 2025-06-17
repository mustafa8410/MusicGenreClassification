package com.example.MusicGenreClassification.repository;

import com.example.MusicGenreClassification.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername (String username);
}
